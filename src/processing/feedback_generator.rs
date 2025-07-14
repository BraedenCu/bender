use std::sync::mpsc::{Receiver, Sender};
use anyhow::Result;
use ort::{session::Session, value::Value};
use tokenizers::Tokenizer;

use crate::{
    core::model_manager::ModelManager,
    model::intent::Intent
};

use super::classifier::ClassificationFailureReason;

pub struct ONNXTextGenerator {
    session: Option<Session>,
    manager: ModelManager,
    tokenizer: Option<Tokenizer>,
}

impl ONNXTextGenerator {
    pub fn new() -> Result<Self> {
        let manager = ModelManager::new()?;
        let session = Self::load_onnx_model(&manager);
        let tokenizer = Self::load_tokenizer(&manager);
        
        println!("DEBUG: ONNXTextGenerator::new() - tokenizer loaded: {}", tokenizer.is_some());
        
        Ok(ONNXTextGenerator {
            session,
            manager,
            tokenizer,
        })
    }
    
    fn load_tokenizer(manager: &ModelManager) -> Option<Tokenizer> {
        let tokenizer_path = manager.get_tokenizer_path();
        
        if !tokenizer_path.exists() {
            println!("Tokenizer not found at: {}", tokenizer_path.display());
            return None;
        }
        
        match Tokenizer::from_file(&tokenizer_path) {
            Ok(tokenizer) => {
                println!("Successfully loaded tokenizer from: {}", tokenizer_path.display());
                Some(tokenizer)
            }
            Err(e) => {
                eprintln!("Failed to load tokenizer: {}", e);
                None
            }
        }
    }
    
    fn load_onnx_model(manager: &ModelManager) -> Option<Session> {
        let model_info = ModelManager::get_text_generation_model_info();
        let model_path = manager.get_model_path(&model_info.name);
        
        if !model_path.exists() {
            println!("ONNX text generation model not found, will use classification-based responses");
            return None;
        }
        
        match Session::builder()
            .unwrap()
            .commit_from_file(&model_path)
        {
            Ok(session) => {
                println!("Successfully loaded ONNX text generation model");
                
                // DEBUG: Print model input/output info
                println!("DEBUG: Text gen model inputs: {:?}", session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
                println!("DEBUG: Text gen model outputs: {:?}", session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());
                
                Some(session)
            }
            Err(e) => {
                eprintln!("Failed to load ONNX text generation model: {}, will use classification-based responses", e);
                None
            }
        }
    }
    
    pub fn generate_with_onnx(&mut self, question: &str) -> Result<String, String> {
        println!("DEBUG: generate_with_onnx called with: '{}'", question);
        
        // First check if we have both session and tokenizer
        println!("DEBUG: session available: {}", self.session.is_some());
        println!("DEBUG: tokenizer available: {}", self.tokenizer.is_some());
        
        if self.session.is_none() {
            return Err("No ONNX text generation session available".to_string());
        }
        if self.tokenizer.is_none() {
            return Err("No tokenizer available".to_string());
        }
        
        println!("DEBUG: Session and tokenizer available, starting tokenization");
        let input_tokens = {
            let tokenizer = self.tokenizer.as_ref().unwrap();
            Self::tokenize_with_tokenizer(question, tokenizer)?
        };
        println!("DEBUG: Input tokens: {:?}", input_tokens);
        
        if input_tokens.is_empty() {
            return Err("Failed to tokenize input".to_string());
        }
        
        println!("DEBUG: Calling generate_tokens");
        let generated_tokens = {
            let session = self.session.as_mut().unwrap();
            Self::generate_tokens(session, &input_tokens, 20)  // Reduced from 50 to 20
                .ok_or("Failed to generate tokens with ONNX model")?
        };
        
        println!("DEBUG: Generated {} tokens: {:?}", generated_tokens.len(), generated_tokens);
        let response = {
            let tokenizer = self.tokenizer.as_ref().unwrap();
            Self::detokenize_with_tokenizer(&generated_tokens, tokenizer)?
        };
        let cleaned_response = Self::clean_response(&response, question);
        
        println!("DEBUG: Final response: '{}'", cleaned_response);
        Ok(cleaned_response)
    }
    
    fn tokenize_with_tokenizer(text: &str, tokenizer: &Tokenizer) -> Result<Vec<i32>, String> {
        let encoding = tokenizer.encode(text, false)
            .map_err(|e| format!("Failed to encode text: {}", e))?;
        
        let tokens: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        println!("DEBUG: Tokenized '{}' to {} tokens: {:?}", text, tokens.len(), &tokens[..tokens.len().min(10)]);
        
        Ok(tokens)
    }
    
    fn detokenize_with_tokenizer(tokens: &[i32], tokenizer: &Tokenizer) -> Result<String, String> {
        let token_ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        
        let decoded = tokenizer.decode(&token_ids, true)
            .map_err(|e| format!("Failed to decode tokens: {}", e))?;
        
        println!("DEBUG: Detokenized {} tokens to: '{}'", tokens.len(), decoded);
        Ok(decoded)
    }
    
    fn generate_tokens(session: &mut Session, input_tokens: &[i32], max_new_tokens: usize) -> Option<Vec<i32>> {
        println!("DEBUG: generate_tokens called with {} input tokens, max_new_tokens: {}", input_tokens.len(), max_new_tokens);
        
        // Use only the actual input tokens, no padding
        let mut tokens = input_tokens.to_vec();
        
        for i in 0..max_new_tokens {
            if i % 10 == 0 {  // Reduce debug spam
                println!("DEBUG: Generation step {}/{}", i + 1, max_new_tokens);
            }
            
            // Prepare input for the model
            let input_ids: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
            let batch_size = 1;
            let seq_len = input_ids.len();
            
            // Create attention_mask (all 1s for no padding)
            let attention_mask: Vec<i64> = vec![1i64; seq_len];
            
            // Create ONNX inputs
            let input_ids_value = Value::from_array(([batch_size, seq_len], input_ids)).ok()?;
            let attention_mask_value = Value::from_array(([batch_size, seq_len], attention_mask)).ok()?;
            
            // Run the model
            let outputs = match session.run(ort::inputs![
                "input_ids" => input_ids_value,
                "attention_mask" => attention_mask_value
            ]) {
                Ok(outputs) => outputs,
                Err(e) => {
                    println!("DEBUG: ONNX text generation inference failed with error: {}", e);
                    return None;
                }
            };
            
            // Get the logits from the output
            // The output should be logits with shape [batch_size, sequence_length, vocab_size]
            if let Ok(logits_tensor) = outputs["logits"].try_extract_tensor::<f32>() {
                let (_shape, logits_data) = logits_tensor;
                
                // Get the logits for the last token
                let vocab_size = 50257; // GPT-2 vocab size
                let last_token_logits_start = logits_data.len() - vocab_size;
                let last_token_logits = &logits_data[last_token_logits_start..];
                
                // Sample the next token
                let next_token = Self::sample_token(last_token_logits);
                
                tokens.push(next_token);
                
                // Stop if we hit the end token (50256 for GPT-2)
                if next_token == 50256 {
                    println!("DEBUG: Hit end token, stopping generation");
                    break;
                }
                
                // Also stop if we hit certain other stop conditions
                if next_token == 0 || (i > 5 && next_token == tokens[tokens.len()-2]) {
                    println!("DEBUG: Hit stop condition, stopping generation");
                    break;
                }
            } else {
                println!("DEBUG: Failed to extract logits tensor");
                break;
            }
        }
        
        println!("DEBUG: Generation complete, total tokens: {}", tokens.len());
        Some(tokens)
    }
    
    fn sample_token(logits: &[f32]) -> i32 {
        // Improved sampling to avoid invalid tokens
        let temperature = 0.8;
        
        // Apply temperature
        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
        
        // Find top-k tokens, filtering out potentially problematic ones
        let top_k = 40;
        let mut indexed_logits: Vec<(usize, f32)> = scaled_logits.iter().enumerate()
            .filter(|(i, _)| {
                // Filter out some potentially problematic token ranges
                let token_id = *i;
                // GPT-2 vocabulary: avoid very high token IDs that might be unused
                token_id < 50000 && token_id > 3  // Avoid very low tokens (padding, etc.)
            })
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_logits.truncate(top_k);
        
        if indexed_logits.is_empty() {
            return 50256; // GPT-2 end token as fallback
        }
        
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
    
    fn clean_response(response: &str, _question: &str) -> String {
        // Clean up the generated response
        let mut cleaned = response.trim().to_string();
        
        // Remove any remaining control characters or unwanted tokens
        cleaned = cleaned.replace("<|endoftext|>", "");
        cleaned = cleaned.replace("<|startoftext|>", "");
        
        // Remove various unwanted artifacts
        cleaned = cleaned.replace("[unused", "");
        cleaned = cleaned.replace("unused]", "");
        cleaned = cleaned.replace("[PAD]", "");
        cleaned = cleaned.replace("[UNK]", "");
        
        // Remove excessive whitespace and clean up
        cleaned = cleaned.split_whitespace()
            .filter(|word| !word.contains("unused") && !word.contains('[') && !word.contains(']'))
            .collect::<Vec<_>>()
            .join(" ");
        
        // Basic cleanup
        cleaned = cleaned.trim().to_string();
        
        // Ensure proper ending
        if !cleaned.ends_with('.') && !cleaned.ends_with('!') && !cleaned.ends_with('?') {
            cleaned.push('.');
        }
        
        // If response is too short or empty, provide a basic response
        if cleaned.len() < 3 {
            "I understand your request.".to_string()
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
        ClassificationFailureReason::UnsupportedInstruction => "Assistant: I apologize, but I don't understand what you're asking me to do. Could you please rephrase your request?",
        ClassificationFailureReason::UnrecognizedInstruction => "Assistant: I'm sorry, I didn't recognize your request. How can I help you today?",
        ClassificationFailureReason::Unknown => "Assistant: I apologize, but something went wrong processing your request. Please try again.",
    };
    
    match text_generator.generate_with_onnx(error_prompt) {
        Ok(response) => {
            // Clean up the response to remove "Assistant:" prefix if present
            let cleaned = response.strip_prefix("Assistant:").unwrap_or(&response).trim();
            cleaned.to_string()
        },
        Err(e) => {
            eprintln!("ONNX text generation failed: {}. Using basic error response.", e);
            match reason {
                ClassificationFailureReason::UnsupportedInstruction => "I don't understand what you're asking me to do.".to_string(),
                ClassificationFailureReason::UnrecognizedInstruction => "I didn't recognize your request.".to_string(),
                ClassificationFailureReason::Unknown => "Something went wrong processing your request.".to_string(),
            }
        }
    }
}

fn generate_onnx_response(intent: Intent, text_generator: &mut ONNXTextGenerator) -> String {
    let prompt = match intent {
        Intent::Command(ref command) => format!("Human: Please {} the {} in the {}\nAssistant: I'll help you with that.", 
            command.action.to_string(), 
            command.subject.to_string(), 
            command.location),
        Intent::Question(ref question) => format!("Human: {}\nAssistant: Let me help you with that.", question),
    };
    
    match text_generator.generate_with_onnx(&prompt) {
        Ok(response) => {
            // Clean up the response
            let cleaned = response.strip_prefix("Human:")
                .or_else(|| response.strip_prefix("Assistant:"))
                .unwrap_or(&response)
                .trim();
            
            // Remove any duplicate conversation markers
            let cleaned = cleaned.replace("Human:", "")
                .replace("Assistant:", "")
                .trim()
                .to_string();
            
            if cleaned.is_empty() {
                "I understand your request.".to_string()
            } else {
                cleaned
            }
        },
        Err(e) => {
            eprintln!("ONNX text generation failed: {}. Using classification-based response.", e);
            // Provide a basic response based on the intent
            match intent {
                Intent::Command(ref command) => {
                    format!("I understand you want to {} the {} in the {}.", 
                        command.action.to_string(), 
                        command.subject.to_string(), 
                        command.location)
                },
                Intent::Question(ref question) => {
                    format!("I received your question about: {}.", question)
                }
            }
        }
    }
}

