use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Result, anyhow};
use reqwest;
use futures_util::StreamExt;
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub size_mb: f64,
    pub description: String,
}

pub struct ModelManager {
    models_dir: PathBuf,
    client: reqwest::Client,
}

impl ModelManager {
    pub fn new() -> Result<Self> {
        let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
        
        // Create models directory if it doesn't exist
        if !models_dir.exists() {
            fs::create_dir_all(&models_dir)?;
        }
        
        let client = reqwest::Client::new();
        
        Ok(ModelManager {
            models_dir,
            client,
        })
    }
    
    pub async fn ensure_models_downloaded(&self) -> Result<()> {
        let classification_model = Self::get_classification_model_info();
        let text_gen_model = Self::get_text_generation_model_info();
        
        println!("Checking if ONNX models are available...");
        
        let mut classification_available = false;
        let mut text_generation_available = false;
        
        // Download classification model if not present
        if !self.get_model_path(&classification_model.name).exists() {
            println!("Downloading classification model: {}", classification_model.name);
            match self.download_model(&classification_model).await {
                Ok(()) => {
                    println!("Successfully downloaded classification model");
                    classification_available = true;
                }
                Err(e) => {
                    println!("Failed to download classification model: {}", e);
                }
            }
        } else {
            println!("Classification model {} already exists", classification_model.name);
            classification_available = true;
        }
        
        // Download text generation model if not present
        if !self.get_model_path(&text_gen_model.name).exists() {
            println!("Downloading text generation model: {}", text_gen_model.name);
            match self.download_model(&text_gen_model).await {
                Ok(()) => {
                    println!("Successfully downloaded text generation model");
                    text_generation_available = true;
                }
                Err(e) => {
                    println!("Failed to download text generation model: {}", e);
                    self.check_manual_model_placement().ok();
                }
            }
        } else {
            println!("Text generation model {} already exists", text_gen_model.name);
            text_generation_available = true;
        }
        
        // Require at least the classification model to be available
        if !classification_available {
            return Err(anyhow!("Classification model is required but could not be downloaded"));
        }
        
        if !text_generation_available {
            println!("Text generation model not available - system will use classification-based responses");
        }
        
        println!("ONNX models initialization complete!");
        Ok(())
    }
    
    pub fn get_model_path(&self, model_name: &str) -> PathBuf {
        self.models_dir.join(format!("{}.onnx", model_name))
    }
    
    pub fn get_tokenizer_path(&self) -> PathBuf {
        self.models_dir.join("tokenizer.json")
    }
    
    pub fn is_model_available(&self, model_name: &str) -> bool {
        self.get_model_path(model_name).exists()
    }
    
    pub async fn download_model(&self, model_info: &ModelInfo) -> Result<()> {
        let model_path = self.get_model_path(&model_info.name);
        
        if model_path.exists() {
            println!("Model {} already exists", model_info.name);
            if !model_info.sha256.is_empty() {
                println!("Verifying integrity...");
                if self.verify_model_integrity(&model_path, &model_info.sha256)? {
                    println!("Model {} is valid and ready to use", model_info.name);
                    return Ok(());
                } else {
                    println!("Model {} failed integrity check, re-downloading...", model_info.name);
                    fs::remove_file(&model_path)?;
                }
            } else {
                println!("Model {} is ready to use (skipping integrity check)", model_info.name);
                return Ok(());
            }
        }
        
        println!("Downloading model {} ({:.1} MB)...", model_info.name, model_info.size_mb);
        
        let response = self.client.get(&model_info.url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Failed to download model: HTTP {}", response.status()));
        }
        
        let mut file = tokio::fs::File::create(&model_path).await?;
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            downloaded += chunk.len() as u64;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
            
            // Show progress for large files
            if downloaded % (1024 * 1024) == 0 {
                println!("Downloaded {} MB...", downloaded / (1024 * 1024));
            }
        }
        
        println!("Model {} downloaded successfully ({} MB)", model_info.name, downloaded / (1024 * 1024));
        
        // Skip SHA256 verification if hash is empty
        if !model_info.sha256.is_empty() {
            // Would verify here, but we're skipping for now
        }
        
        Ok(())
    }
    
    pub fn verify_model_integrity(&self, model_path: &Path, expected_sha256: &str) -> Result<bool> {
        let contents = fs::read(model_path)?;
        let mut hasher = Sha256::new();
        hasher.update(&contents);
        let computed_hash = hex::encode(hasher.finalize());
        
        Ok(computed_hash == expected_sha256)
    }
    
    pub fn get_classification_model_info() -> ModelInfo {
        ModelInfo {
            name: "all-MiniLM-L6-v2".to_string(),
            url: "https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/model.onnx".to_string(),
            sha256: "".to_string(), // We'll skip verification for now
            size_mb: 90.9,
            description: "ONNX sentence transformer for text embeddings (384 dimensions)".to_string(),
        }
    }
    
    pub fn get_text_generation_model_info() -> ModelInfo {
        ModelInfo {
            name: "decoder_model".to_string(),
            url: "https://huggingface.co/optimum/gpt2/resolve/main/onnx/decoder_model.onnx".to_string(),
            sha256: "".to_string(), // We'll skip verification for now
            size_mb: 500.0, // GPT-2 model size
            description: "GPT-2 ONNX model for conversational text generation".to_string(),
        }
    }

    /// Support for manually placed models
    pub fn check_manual_model_placement(&self) -> Result<()> {
        let text_gen_model = Self::get_text_generation_model_info();
        let model_path = self.get_model_path(&text_gen_model.name);
        
        if !model_path.exists() {
            println!("\n📋 MANUAL MODEL DOWNLOAD INSTRUCTIONS:");
            println!("   If automatic download fails, you can manually download the text generation model:");
            println!("   1. Go to: https://huggingface.co/optimum/gpt2");
            println!("   2. Download: onnx/decoder_model.onnx");
            println!("   3. Place it at: {:?}", model_path);
            println!("   4. Restart the application\n");
            println!("   Alternative: You can place any compatible ONNX text generation model in the models/ directory");
        }
        
        Ok(())
    }
}

// Convenience function for use in main
pub async fn ensure_models_available() -> Result<()> {
    let manager = ModelManager::new()?;
    manager.ensure_models_downloaded().await
}

