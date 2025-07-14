use std::sync::mpsc::{Receiver, Sender};
use anyhow::Result;
use ort::{session::Session, value::Value};
use ndarray::Array2;

use crate::{
    core::model_manager::ModelManager,
    model::intent::Intent
};

use super::classifier::ClassificationFailureReason;

pub struct ONNXTextGenerator {
    session: Option<Session>,
    manager: ModelManager,
}

impl ONNXTextGenerator {
    pub fn new() -> Result<Self> {
        let manager = ModelManager::new()?;
        let session = Self::load_onnx_model(&manager);
        
        Ok(ONNXTextGenerator {
            session,
            manager,
        })
    }
    
    fn load_onnx_model(manager: &ModelManager) -> Option<Session> {
        let model_info = ModelManager::get_text_generation_model_info();
        let model_path = manager.get_model_path(&model_info.name);
        
        if !model_path.exists() {
            println!("ONNX text generation model not found, system requires models to function");
            return None;
        }
        
        match Session::builder()
            .unwrap()
            .commit_from_file(&model_path)
        {
            Ok(session) => {
                println!("Successfully loaded ONNX text generation model");
                Some(session)
            }
            Err(e) => {
                eprintln!("Failed to load ONNX text generation model: {}, system requires models to function", e);
                None
            }
        }
    }
    
    pub fn generate_with_onnx(&mut self, question: &str) -> Result<String, String> {
        let session = self.session.as_mut().ok_or("No ONNX text generation session available. System requires ONNX models to function.")?;
        
        // Create a simple prompt for GPT-2
        let prompt = format!("Human: {}\nAssistant:", question.trim());
        
        // Simple tokenization for GPT-2
        let tokens = Self::simple_tokenize(&prompt);
        
        // Generate response
        let generated_tokens = Self::generate_tokens(session, &tokens, 30).ok_or("Failed to generate tokens with ONNX model")?;
        
        // Convert back to text
        let response = Self::detokenize(&generated_tokens);
        
        // Clean up response
        let cleaned_response = Self::clean_response(&response, question);
        
        println!("ONNX generated response for '{}': '{}'", question, cleaned_response);
        
        Ok(cleaned_response)
    }
    
    fn simple_tokenize(text: &str) -> Vec<i32> {
        // Very basic tokenization for GPT-2
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = vec![50256]; // GPT-2 BOS token
        
        for word in words {
            // Simple hash-based token ID generation (not proper tokenization)
            let token_id = (word.chars().map(|c| c as u32).sum::<u32>() % 50000) as i32 + 100;
            tokens.push(token_id);
        }
        
        tokens
    }
    
    fn generate_tokens(session: &mut Session, input_tokens: &[i32], max_new_tokens: usize) -> Option<Vec<i32>> {
        let mut current_tokens = input_tokens.to_vec();
        let mut generated_tokens = Vec::new();
        
        for _ in 0..max_new_tokens {
            // Prepare input for the model (use last 512 tokens as context)
            let context_length = current_tokens.len().min(512);
            let input_slice = &current_tokens[current_tokens.len() - context_length..];
            
            // Create input array for ONNX
            let input_ids = Array2::from_shape_vec((1, input_slice.len()), input_slice.to_vec()).ok()?;
            let input_ids_i64 = input_ids.mapv(|x| x as i64);
            let shape = input_ids_i64.shape().to_vec();
            let raw_data = input_ids_i64.into_raw_vec();
            
            // Create input value for ONNX
            let input_value = Value::from_array(([shape[0], shape[1]], raw_data)).ok()?;
            
            // Run inference with standard input name
            let outputs = session.run(ort::inputs!["input_ids" => input_value]).ok()?;
            
            // Try to get logits from different possible output names
            let output_names = ["logits", "output", "outputs", "last_hidden_state"];
            
            for output_name in &output_names {
                if let Ok(logits_tensor) = outputs[*output_name].try_extract_tensor::<f32>() {
                    let (_shape, data) = logits_tensor;
                    
                    // Sample next token from the last position
                    if data.len() >= 50000 { // GPT-2 vocab size ~50k
                        let vocab_size = 50000;
                        let last_logits_start = data.len() - vocab_size;
                        let last_logits_slice = &data[last_logits_start..];
                        
                        let next_token = Self::sample_token(last_logits_slice);
                        
                        // Check for GPT-2 end tokens
                        if next_token == 50256 || next_token == 0 { // EOS or PAD tokens
                            break;
                        }
                        
                        generated_tokens.push(next_token);
                        current_tokens.push(next_token);
                        break;
                    }
                }
            }
        }
        
        Some(generated_tokens)
    }
    
    fn sample_token(logits: &[f32]) -> i32 {
        // Simple sampling: find the most likely token with some randomness
        let temperature = 0.7;
        
        // Apply temperature
        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        
        // Find top-k tokens
        let top_k = 50;
        let mut indexed_logits: Vec<(usize, f32)> = scaled_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_logits.truncate(top_k);
        
        // Convert to probabilities
        let max_logit = indexed_logits[0].1;
        let exp_logits: Vec<f32> = indexed_logits.iter().map(|(_, logit)| (logit - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        
        if sum_exp <= 0.0 {
            return indexed_logits[0].0 as i32;
        }
        
        let probabilities: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();
        
        // Sample from distribution
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let random_value: f32 = rng.gen();
        let mut cumulative = 0.0;
        
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value < cumulative {
                return indexed_logits[i].0 as i32;
            }
        }
        
        // Fallback to most likely
        indexed_logits[0].0 as i32
    }
    
    fn detokenize(tokens: &[i32]) -> String {
        // Very basic detokenization - should use proper Llama tokenizer
        let words: Vec<String> = tokens.iter().map(|&token_id| {
            format!("token{}", token_id % 1000)
        }).collect();
        
        words.join(" ")
    }
    
    fn clean_response(response: &str, _question: &str) -> String {
        // Clean up the generated response
        let mut cleaned = response.trim().to_string();
        
        // Remove "token" prefixes from our simple detokenizer
        cleaned = cleaned.replace("token", " ");
        
        // Basic cleanup
        cleaned = cleaned.trim().to_string();
        
        // Ensure proper ending
        if !cleaned.ends_with('.') && !cleaned.ends_with('!') && !cleaned.ends_with('?') {
            cleaned.push('.');
        }
        
        // If response is too short or nonsensical, indicate model limitation
        if cleaned.len() < 10 {
            "I need a proper tokenizer to generate better responses.".to_string()
        } else {
            cleaned
        }
    }
}

pub fn main(intent_rx: Receiver<Result<Intent, ClassificationFailureReason>>, feedback_tx: Sender<String>) -> Result<()> {    
    let mut text_generator = ONNXTextGenerator::new()?;
    
    while let Ok(result) = intent_rx.recv() { 
        let message = match result {
            Ok(intent) => generate_onnx_response(intent, &mut text_generator),
            Err(error) => generate_onnx_error_response(error, &mut text_generator)
        };
        
        println!("Feedback message: '{}'\n", message);
        if feedback_tx.send(message).is_err() {
            break;
        }
    }
    
    Ok(())
}

fn generate_onnx_error_response(reason: ClassificationFailureReason, text_generator: &mut ONNXTextGenerator) -> String {
    let error_prompt = match reason {
        ClassificationFailureReason::UnsupportedInstruction => "I don't understand what you're asking me to do",
        ClassificationFailureReason::UnrecognizedInstruction => "I didn't recognize your request",
        ClassificationFailureReason::Unknown => "Something went wrong processing your request",
    };
    
    match text_generator.generate_with_onnx(error_prompt) {
        Ok(response) => response,
        Err(e) => {
            eprintln!("ONNX text generation failed: {}. System requires ONNX models to function.", e);
            "I cannot respond because the required ONNX models are not available.".to_string()
        }
    }
}

fn generate_onnx_response(intent: Intent, text_generator: &mut ONNXTextGenerator) -> String {
    let prompt = match intent {
        Intent::Command(ref command) => format!("I need to {} the {} in the {}", 
            command.action.to_string(), 
            command.subject.to_string(), 
            command.location),
        Intent::Question(question) => question,
    };
    
    match text_generator.generate_with_onnx(&prompt) {
        Ok(response) => response,
        Err(e) => {
            eprintln!("ONNX text generation failed: {}. System requires ONNX models to function.", e);
            "I cannot respond because the required ONNX models are not available.".to_string()
        }
    }
}

