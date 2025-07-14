use std::sync::mpsc::{Receiver, Sender};
use ort::{session::Session, value::Value};
use tokenizers::Tokenizer;
use ndarray::Array1;
use anyhow::Result;
use crate::model::command::Command;
use crate::model::intent::Intent;
use crate::model::command_action::{CommandAction, CommandSwitchValue};
use crate::model::command_subject::CommandSubject;
use crate::core::model_manager::ModelManager;
use crate::core::commander::Commander;
use crate::traits::labelable::Labelable;

pub type ClassifierOutput = Result<Intent, ClassificationFailureReason>;

const SCORE_THRESHOLD: f64 = 0.75;

struct ClassificationLabels {
    intents: Vec<String>,
    locations: Vec<String>,
    actions: Vec<String>,
    subjects: Vec<String>
}

#[derive(Debug)]
pub struct Label {
    pub text: String,
    pub score: f64,
}

pub enum ClassificationFailureReason {
    Unknown, UnsupportedInstruction, UnrecognizedInstruction
}

pub struct ONNXClassifier {
    session: Option<Session>,
    tokenizer: Option<Tokenizer>,
    manager: ModelManager,
}

impl ONNXClassifier {
    pub fn new() -> Result<Self> {
        let manager = ModelManager::new()?;
        let (session, tokenizer) = Self::load_onnx_model(&manager);
        
        Ok(ONNXClassifier {
            session,
            tokenizer,
            manager,
        })
    }
    
    fn load_onnx_model(manager: &ModelManager) -> (Option<Session>, Option<Tokenizer>) {
        let model_path = manager.get_model_path("all-MiniLM-L6-v2");
        
        if !model_path.exists() {
            println!("ONNX model not found at {:?}", model_path);
            return (None, None);
        }
        
        // Load ONNX session
        let session = Session::builder()
            .map_err(|e| println!("Failed to create session builder: {}", e))
            .ok()
            .and_then(|builder| {
                builder.commit_from_file(&model_path)
                    .map_err(|e| println!("Failed to load model from {:?}: {}", model_path, e))
                    .ok()
            });
        
        // Load tokenizer
        let tokenizer_path = manager.get_tokenizer_path();
        let tokenizer = if tokenizer_path.exists() {
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| println!("Failed to load tokenizer: {}", e))
                .ok()
        } else {
            println!("Tokenizer not found at {:?}", tokenizer_path);
            None
        };
        
        if session.is_some() && tokenizer.is_some() {
            println!("Successfully loaded ONNX classification model");
            
            // DEBUG: Print model input/output info
            if let Some(ref sess) = session {
                println!("DEBUG: Model inputs: {:?}", sess.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
                println!("DEBUG: Model outputs: {:?}", sess.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());
            }
        } else {
            println!("Failed to load ONNX model or tokenizer");
        }
        
        (session, tokenizer)
    }
    
    pub fn classify_with_onnx(&mut self, instruction: &str, labels: &ClassificationLabels) -> Result<Vec<Label>, String> {
        let session = self.session.as_mut().ok_or("No ONNX session available")?;
        let tokenizer = self.tokenizer.as_ref().ok_or("No tokenizer available")?;
        
        // Get embeddings for the input instruction
        let input_embedding = Self::get_text_embedding(instruction, session, tokenizer).ok_or("Failed to get input embedding")?;
        
        // Get embeddings for all labels
        let mut label_scores = Vec::new();
        
        // Process intent labels
        for intent in &labels.intents {
            if let Some(label_embedding) = Self::get_text_embedding(intent, session, tokenizer) {
                let similarity = cosine_similarity(&input_embedding, &label_embedding);
                label_scores.push(Label {
                    text: intent.clone(),
                    score: similarity,
                });
            }
        }
        
        // Process action labels  
        for action in &labels.actions {
            if let Some(label_embedding) = Self::get_text_embedding(action, session, tokenizer) {
                let similarity = cosine_similarity(&input_embedding, &label_embedding);
                label_scores.push(Label {
                    text: action.clone(),
                    score: similarity,
                });
            }
        }
        
        // Process subject labels
        for subject in &labels.subjects {
            if let Some(label_embedding) = Self::get_text_embedding(subject, session, tokenizer) {
                let similarity = cosine_similarity(&input_embedding, &label_embedding);
                label_scores.push(Label {
                    text: subject.clone(),
                    score: similarity,
                });
            }
        }
        
        // Process location labels
        for location in &labels.locations {
            if let Some(label_embedding) = Self::get_text_embedding(location, session, tokenizer) {
                let similarity = cosine_similarity(&input_embedding, &label_embedding);
                label_scores.push(Label {
                    text: location.clone(),
                    score: similarity,
                });
            }
        }
        
        // Sort by similarity score (descending)
        label_scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(label_scores)
    }
    
    fn get_text_embedding(text: &str, session: &mut Session, tokenizer: &Tokenizer) -> Option<Array1<f32>> {
        println!("DEBUG: Starting get_text_embedding for text: '{}'", text);
        
        // Use the real tokenizer to encode the text
        let encoding = tokenizer.encode(text, true).ok()?;
        
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        
        println!("DEBUG: Tokenized - input_ids len: {}, attention_mask len: {}", input_ids.len(), attention_mask.len());
        
        if input_ids.is_empty() {
            println!("DEBUG: Empty input_ids, returning None");
            return None;
        }
        
        // Convert to the format expected by ONNX Runtime
        let batch_size = 1;
        let sequence_length = input_ids.len();
        
        // Create i64 arrays for ONNX Runtime
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let attention_mask_i64: Vec<i64> = attention_mask.iter().map(|&x| x as i64).collect();
        // Create token_type_ids (all zeros for single sentence)
        let token_type_ids_i64: Vec<i64> = vec![0i64; sequence_length];
        
        // Create Value objects for ONNX Runtime
        let input_ids_value = Value::from_array(([batch_size, sequence_length], input_ids_i64)).ok()?;
        let attention_mask_value = Value::from_array(([batch_size, sequence_length], attention_mask_i64)).ok()?;
        let token_type_ids_value = Value::from_array(([batch_size, sequence_length], token_type_ids_i64)).ok()?;
        
        // Run inference
        println!("DEBUG: About to run ONNX session inference...");
        let outputs = match session.run(ort::inputs![
            "input_ids" => input_ids_value,
            "attention_mask" => attention_mask_value,
            "token_type_ids" => token_type_ids_value
        ]) {
            Ok(outputs) => {
                println!("DEBUG: ONNX inference successful!");
                outputs
            }
            Err(e) => {
                println!("DEBUG: ONNX inference failed with error: {}", e);
                return None;
            }
        };
        
        // DEBUG: Print available output names
        println!("DEBUG: Available model outputs: {:?}", outputs.keys().collect::<Vec<_>>());
        
        // For sentence transformers, we need to extract the pooled output
        // Let's try to get the first output if the specific names don't work
        let output_names = ["last_hidden_state", "pooler_output", "sentence_embedding"];
        
        for output_name in &output_names {
            if let Ok(tensor) = outputs[*output_name].try_extract_tensor::<f32>() {
                println!("DEBUG: Using output name: {}", output_name);
                let (shape, data) = tensor;
                println!("DEBUG: Output shape: {:?}", shape);
                
                // For sentence transformers, we typically do mean pooling
                // The shape is usually [batch_size, sequence_length, hidden_size]
                if shape.len() == 3 && shape[0] == batch_size as i64 && shape[2] == 384 {
                    // Do mean pooling over the sequence length dimension
                    let seq_len = shape[1] as usize;
                    let hidden_size = 384;
                    
                    let mut pooled = vec![0.0f32; hidden_size];
                    
                    for i in 0..seq_len {
                        for j in 0..hidden_size {
                            let idx = i * hidden_size + j;
                            if idx < data.len() {
                                pooled[j] += data[idx];
                            }
                        }
                    }
                    
                    // Average by sequence length
                    for val in pooled.iter_mut() {
                        *val /= seq_len as f32;
                    }
                    
                    // Normalize the embedding
                    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in pooled.iter_mut() {
                            *val /= norm;
                        }
                    }
                    
                    return Some(Array1::from_vec(pooled));
                }
                // If it's already pooled (shape [batch_size, hidden_size])
                else if shape.len() == 2 && shape[0] == batch_size as i64 && shape[1] == 384 {
                    let embedding = data[0..384].to_vec();
                    let embedding_array = Array1::from_vec(embedding);
                    
                    // Normalize
                    let norm = (embedding_array.mapv(|x| x * x).sum()).sqrt();
                    if norm > 0.0 {
                        return Some(embedding_array / norm);
                    } else {
                        return Some(embedding_array);
                    }
                }
            }
        }
        
        // If named outputs don't work, try to use the first available output
        if outputs.len() > 0 {
            // Try to get the first output
            let output_key = outputs.keys().next().unwrap();
            println!("DEBUG: Trying first available output: {}", output_key);
            if let Ok(tensor) = outputs[output_key].try_extract_tensor::<f32>() {
                let (shape, data) = tensor;
                println!("DEBUG: First output shape: {:?}, data len: {}", shape, data.len());
                
                // Handle different potential shapes more flexibly
                match shape.len() {
                    3 => {
                        // [batch_size, sequence_length, hidden_size]
                        if shape[0] == batch_size as i64 {
                            let seq_len = shape[1] as usize;
                            let hidden_size = shape[2] as usize;
                            
                            let mut pooled = vec![0.0f32; hidden_size];
                            
                            for i in 0..seq_len {
                                for j in 0..hidden_size {
                                    let idx = i * hidden_size + j;
                                    if idx < data.len() {
                                        pooled[j] += data[idx];
                                    }
                                }
                            }
                            
                            // Average by sequence length
                            for val in pooled.iter_mut() {
                                *val /= seq_len as f32;
                            }
                            
                            // Normalize the embedding
                            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for val in pooled.iter_mut() {
                                    *val /= norm;
                                }
                            }
                            
                            return Some(Array1::from_vec(pooled));
                        }
                    }
                    2 => {
                        // [batch_size, hidden_size]
                        if shape[0] == batch_size as i64 {
                            let hidden_size = shape[1] as usize;
                            let embedding = data[0..hidden_size].to_vec();
                            let embedding_array = Array1::from_vec(embedding);
                            
                            // Normalize
                            let norm = (embedding_array.mapv(|x| x * x).sum()).sqrt();
                            if norm > 0.0 {
                                return Some(embedding_array / norm);
                            } else {
                                return Some(embedding_array);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        
        None
    }
}

fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.mapv(|x| x * x).sum().sqrt();
    let norm_b: f32 = b.mapv(|x| x * x).sum().sqrt();
    
    if norm_a > 0.0 && norm_b > 0.0 {
        (dot_product / (norm_a * norm_b)) as f64
    } else {
        0.0
    }
}

pub fn main(command_rx: Receiver<String>, intent_tx: Sender<ClassifierOutput>) -> Result<()> {
    let commander = Commander::new();
    let labels = build_labels(&commander);
    
    let mut classifier = ONNXClassifier::new()?;
    
    while let Ok(instruction) = command_rx.recv() {
        let classification_result = match classifier.classify_with_onnx(&instruction, &labels) {
            Ok(onnx_result) => onnx_result,
            Err(e) => {
                eprintln!("ONNX classification failed: {}. System requires ONNX models to function.", e);
                // Return failure without fallback
                if intent_tx.send(Err(ClassificationFailureReason::Unknown)).is_err() {
                    break;
                }
                continue;
            }
        };
        
        let (intent, score) = intent_from_classification(&instruction, &classification_result, &labels);
        let result: Result<Intent, ClassificationFailureReason>;

        if score < SCORE_THRESHOLD {
            println!("Instruction '{}'\nScore {} with classification: {:?}\n", instruction, score, classification_result);
            result = Err(ClassificationFailureReason::UnrecognizedInstruction);
        } else if let Intent::Command(ref command) = intent {
            if commander.supports_command(command) {
                println!("Instruction '{}'\nExecuting {:?}", instruction, intent);
                result = Ok(intent);
            } else {
                result = Err(ClassificationFailureReason::UnsupportedInstruction);
            }
        } else if let Intent::Question(_) = intent {
            result = Ok(intent);
        } else {
            println!("No suitable command for '{}'\n", instruction);
            result = Err(ClassificationFailureReason::UnsupportedInstruction);
        }

        if intent_tx.send(result).is_err() {
            break;
        }
    }

    Ok(())
}



fn build_labels(commander: &Commander) -> ClassificationLabels {
    ClassificationLabels {
        intents: Intent::labels(),
        locations: commander.locations.clone(),
        actions: CommandAction::labels(),
        subjects: CommandSubject::labels()
    }
}

fn intent_from_classification(instruction: &str, model_output: &Vec<Label>, data: &ClassificationLabels) -> (Intent, f64) {
    let mut action: (f64, usize) = (0.0, 0);
    let mut location: (f64, usize) = (0.0, 0);
    let mut subject: (f64, usize) = (0.0, 0);

    for (i, label) in model_output.iter().enumerate() {
        let score = label.score;

        // Check for questions
        if data.intents.contains(&label.text) && score > SCORE_THRESHOLD {
            if Intent::is_label_question(&label.text) {
                return (Intent::Question(instruction.to_string()), score);
            }
        } else if data.actions.contains(&label.text) && score > action.0 {
            action = (score, i);
        } else if data.subjects.contains(&label.text) && score > subject.0 {
            subject = (score, i);
        } else if data.locations.contains(&label.text) && score > location.0 {
            location = (score, i);
        }
    }

    if action.0 > 0.0 && subject.0 > 0.0 && location.0 > 0.0 {
        let score = action.0.min(location.0).min(subject.0);
        let command = Command {
            location: model_output[location.1].text.clone(),
            action: model_output[action.1].text.parse::<CommandAction>().unwrap_or(CommandAction::Switch(CommandSwitchValue::Off)),
            subject: model_output[subject.1].text.parse::<CommandSubject>().unwrap_or(CommandSubject::Light)
        };
        let intent = Intent::Command(command);
        (intent, score)
    } else {
        // Default to a question if we can't classify properly
        (Intent::Question(instruction.to_string()), 0.6)
    }
}