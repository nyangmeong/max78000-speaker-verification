# MAX78000 Speaker Verification

Text-Independent Speaker Verification on Embedded Devices

## Overview

This project implements **text-independent speaker verification (TI-SV)** optimized for resource-constrained devices like the MAX78000 microcontroller. The system is designed to authenticate speakers using short utterances without requiring specific spoken text, making it practical for embedded applications.

### Key Characteristics

- **Text-Independent**: Speaker authentication works with any spoken utterance - no pre-defined text required
- **Short Utterance**: Efficient verification using brief speech segments (1.3 seconds)
- **Embedded-Optimized**: Designed for low-power microcontroller deployment with limited memory and compute
- **INT8 Quantization**: Neural network models quantized to 8-bit precision for MAX78000 execution
- **VoxCeleb1-O Training**: Pre-trained on large-scale speaker discrimination dataset


## Technical Approach

1. **Speaker Embedding**: Neural network extracts speaker-discriminative features from short audio utterances
2. **Similarity Scoring**: Cosine/Euclidean distance between enrollment and test embeddings
3. **Threshold-Based Decision**: Accept/reject based on similarity threshold
4. **Microcontroller Deployment**: Synthesized to MAX78000-compatible C code via ai8x-synthesis

## Requirements & Installation

### Model Synthesis & Deployment
For generating MAX78000-compatible code and deployment guides, follow the official **ai8x-synthesis** documentation:

```bash
# Clone ai8x-synthesis
git clone https://github.com/analogdevicesinc/ai8x-synthesis.git
cd ai8x-synthesis
pip install -e .
```

Refer to [ai8x-synthesis GitHub](https://github.com/analogdevicesinc/ai8x-synthesis) for complete synthesis workflow, parameter tuning, and embedded deployment guidelines.


### Deployment to MAX78000

Use **ai8x-synthesis** to convert the trained model to MAX78000 C code. Follow the [ai8x-synthesis documentation](https://github.com/analogdevicesinc/ai8x-synthesis) for:
- Model quantization parameters
- Embedded code generation
- Board-specific deployment instructions

## Performance Metrics

- **EER (Equal Error Rate)**: Primary metric for speaker verification performance
- **Evaluation Modes**:
  - `seg 1v1`: Short segment comparison (for embedded real-time scenarios)
  - `utt vs 1`: Standard speaker verification benchmark

## References

- [MAX78000 Documentation](https://www.maximintegrated.com/en/products/microcontrollers/MAX78000.html)
- [ai8x-synthesis GitHub](https://github.com/analogdevicesinc/ai8x-synthesis) - Essential for MAX78000 synthesis and deployment
- [VoxCeleb1 Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

## License

MIT License

## Author

Jae-Woo Ahn

