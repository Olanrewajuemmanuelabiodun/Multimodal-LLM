# Multimodal Large Language Models API Documentation

## Overview

This API provides functionality for working with multimodal large language models, specifically using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. The service enables text and image embedding generation, similarity calculations, and cross-modal analysis.

## Base Requirements

```python
pip install matplotlib transformers datasets accelerate sentence-transformers
```

## Core Dependencies

- `transformers` - Hugging Face transformers library
- `torch` - PyTorch for tensor operations
- `PIL` - Python Imaging Library for image processing
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Machine learning utilities

## Quick Start

```python
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from PIL import Image

# Initialize the model
model_id = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
```

## API Reference

### Core Classes

#### CLIPTokenizerFast
Tokenizer for preprocessing text input.

**Methods:**
- `from_pretrained(model_id)` - Load pretrained tokenizer
- `__call__(text, return_tensors="pt")` - Tokenize input text
- `convert_ids_to_tokens(input_ids)` - Convert token IDs back to tokens

#### CLIPProcessor
Processor for preprocessing images.

**Methods:**
- `from_pretrained(model_id)` - Load pretrained processor
- `__call__(text=None, images=image, return_tensors='pt')` - Process images

#### CLIPModel
Main model for generating embeddings.

**Methods:**
- `from_pretrained(model_id)` - Load pretrained model
- `get_text_features(**inputs)` - Generate text embeddings
- `get_image_features(pixel_values)` - Generate image embeddings

### API Functions

#### Text Embedding Generation

```python
def generate_text_embedding(text: str) -> torch.Tensor:
    """
    Generate normalized text embedding for input text.
    
    Args:
        text (str): Input text to embed
        
    Returns:
        torch.Tensor: Normalized text embedding vector
        
    Example:
        >>> embedding = generate_text_embedding("image of a cat")
        >>> print(embedding.shape)  # torch.Size([1, 512])
    """
    inputs = tokenizer(text, return_tensors="pt")
    text_embedding = model.get_text_features(**inputs)
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding
```

#### Image Embedding Generation

```python
def generate_image_embedding(image: Image.Image) -> torch.Tensor:
    """
    Generate normalized image embedding for input image.
    
    Args:
        image (PIL.Image.Image): Input image to embed
        
    Returns:
        torch.Tensor: Normalized image embedding vector
        
    Example:
        >>> from PIL import Image
        >>> image = Image.open("path/to/image.jpg")
        >>> embedding = generate_image_embedding(image)
        >>> print(embedding.shape)  # torch.Size([1, 512])
    """
    processed_image = processor(
        text=None, images=image, return_tensors='pt'
    )['pixel_values']
    image_embedding = model.get_image_features(processed_image)
    image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
    return image_embedding
```

#### Similarity Calculation

```python
def calculate_similarity(text_embedding: torch.Tensor, 
                        image_embedding: torch.Tensor) -> float:
    """
    Calculate cosine similarity between text and image embeddings.
    
    Args:
        text_embedding (torch.Tensor): Normalized text embedding
        image_embedding (torch.Tensor): Normalized image embedding
        
    Returns:
        float: Cosine similarity score (-1 to 1)
        
    Example:
        >>> text_emb = generate_text_embedding("a dog")
        >>> image_emb = generate_image_embedding(dog_image)
        >>> similarity = calculate_similarity(text_emb, image_emb)
        >>> print(f"Similarity: {similarity:.3f}")
    """
    text_emb_np = text_embedding.detach().cpu().numpy()
    image_emb_np = image_embedding.detach().cpu().numpy()
    return float(text_emb_np @ image_emb_np.T)
```

#### Batch Processing

```python
def process_multiple_images(images: List[Image.Image], 
                           captions: List[str]) -> np.ndarray:
    """
    Process multiple images and captions to generate similarity matrix.
    
    Args:
        images (List[PIL.Image.Image]): List of images to process
        captions (List[str]): List of text captions
        
    Returns:
        np.ndarray: Similarity matrix (images x captions)
        
    Example:
        >>> images = [Image.open(f"image_{i}.jpg") for i in range(3)]
        >>> captions = ["a cat", "a dog", "a bird"]
        >>> sim_matrix = process_multiple_images(images, captions)
        >>> print(sim_matrix.shape)  # (3, 3)
    """
    # Generate image embeddings
    image_embeddings = []
    for image in images:
        image_processed = processor(images=image, return_tensors="pt")["pixel_values"]
        image_embedding = model.get_image_features(image_processed).detach().cpu().numpy()[0]
        image_embeddings.append(image_embedding)
    image_embeddings = np.array(image_embeddings)
    
    # Generate text embeddings
    text_embeddings = []
    for caption in captions:
        inputs = tokenizer(caption, return_tensors="pt")
        text_emb = model.get_text_features(**inputs).detach().cpu().numpy()[0]
        text_embeddings.append(text_emb)
    text_embeddings = np.array(text_embeddings)
    
    # Calculate similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(image_embeddings, text_embeddings)
```

### Visualization Functions

#### Similarity Matrix Visualization

```python
def visualize_similarity_matrix(similarity_matrix: np.ndarray,
                               images: List[Image.Image],
                               captions: List[str],
                               save_path: str = "similarity_matrix.png") -> None:
    """
    Create and save a visualization of the similarity matrix.
    
    Args:
        similarity_matrix (np.ndarray): Similarity scores matrix
        images (List[PIL.Image.Image]): List of images
        captions (List[str]): List of captions
        save_path (str): Path to save the visualization
        
    Example:
        >>> sim_matrix = process_multiple_images(images, captions)
        >>> visualize_similarity_matrix(sim_matrix, images, captions)
    """
    plt.figure(figsize=(20, 14))
    plt.imshow(similarity_matrix, cmap='viridis')
    
    # Set labels and layout
    plt.yticks(range(len(captions)), captions, fontsize=18)
    plt.xticks([])
    
    # Add images to the plot
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    
    # Add similarity scores as text
    for x in range(similarity_matrix.shape[1]):
        for y in range(similarity_matrix.shape[0]):
            plt.text(x, y, f"{similarity_matrix[y, x]:.2f}", 
                    ha="center", va="center", size=30)
    
    # Clean up the plot
    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)
    
    plt.xlim([-0.5, len(captions) - 0.5])
    plt.ylim([len(captions) + 0.5, -2])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

## Model Information

### Supported Models

- **openai/clip-vit-base-patch32**: Base CLIP model with Vision Transformer backbone
- **Model Architecture**: Vision Transformer (ViT) + Text Transformer
- **Embedding Dimension**: 512
- **Supported Image Formats**: PNG, JPEG, RGB
- **Maximum Text Length**: 77 tokens

### Performance Characteristics

- **Image Processing**: ~100ms per image
- **Text Processing**: ~10ms per text
- **Memory Usage**: ~2GB for base model
- **Embedding Size**: 512 dimensions per embedding

## Error Handling

### Common Errors

1. **Image Format Error**
   ```python
   # Ensure images are in RGB format
   image = Image.open(path).convert("RGB")
   ```

2. **Text Length Error**
   ```python
   # Texts longer than 77 tokens will be truncated
   # Use truncation=True for explicit handling
   inputs = tokenizer(text, return_tensors="pt", truncation=True)
   ```

3. **Memory Error**
   ```python
   # Process large batches in chunks
   # Move tensors to CPU when not needed
   embedding = embedding.detach().cpu()
   ```

## Best Practices

### Performance Optimization

1. **Batch Processing**: Process multiple items together when possible
2. **GPU Usage**: Move model to GPU for faster inference
3. **Memory Management**: Use `.detach().cpu()` to free GPU memory
4. **Caching**: Cache embeddings for frequently used images/texts

### Quality Guidelines

1. **Image Quality**: Use high-resolution, clear images
2. **Text Quality**: Use descriptive, specific captions
3. **Preprocessing**: Ensure consistent image preprocessing
4. **Normalization**: Always normalize embeddings for similarity calculations

## Examples

### Basic Usage Example

```python
from PIL import Image

# Load and process an image
image = Image.open("example.jpg").convert("RGB")
caption = "a beautiful landscape"

# Generate embeddings
text_embedding = generate_text_embedding(caption)
image_embedding = generate_image_embedding(image)

# Calculate similarity
similarity = calculate_similarity(text_embedding, image_embedding)
print(f"Similarity score: {similarity:.3f}")
```

### Batch Processing Example

```python
# Load multiple images and captions
images = [Image.open(f"image_{i}.jpg").convert("RGB") for i in range(3)]
captions = ["image of earth", "image of heart", "image of turkey"]

# Process all at once
similarity_matrix = process_multiple_images(images, captions)

# Visualize results
visualize_similarity_matrix(similarity_matrix, images, captions)
```

## Changelog

### Version 1.0.0
- Initial implementation with CLIP model support
- Text and image embedding generation
- Similarity calculation functions
- Batch processing capabilities
- Visualization tools

## Support

For issues and questions regarding this API, please refer to:
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [OpenAI CLIP Model Card](https://huggingface.co/openai/clip-vit-base-patch32)
- [PyTorch Documentation](https://pytorch.org/docs/)