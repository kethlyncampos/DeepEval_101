# DeepEval 101: LLM Evaluation with Google Gemini

A comprehensive tutorial and implementation of LLM evaluation using DeepEval with Google's Gemini model. This project demonstrates how to evaluate the quality of AI-generated content using multiple metrics including coherence, groundedness, relevance, accuracy, and completeness.

## What is DeepEval?

[DeepEval](https://deepeval.com/docs/getting-started) is a powerful Python framework for evaluating Large Language Models (LLMs). It provides:

- **Automated Evaluation**: Test LLM outputs against various quality metrics
- **Custom Metrics**: Create your own evaluation criteria using natural language
- **Multiple LLM Support**: Works with various LLM providers including OpenAI, Anthropic, Google, and custom models
- **Batch Evaluation**: Evaluate multiple test cases efficiently
- **Visualization**: Generate reports and visualizations of evaluation results

## Project Overview

This repository contains a complete implementation of LLM evaluation using DeepEval with Google's Gemini model. The project evaluates the quality of academic paper abstracts against their corresponding full articles using multiple evaluation metrics.

### Key Features

- **Custom LLM Integration**: Implements Google Gemini 2.5 Flash as the evaluation model
- **Multi-Metric Evaluation**: Evaluates content across 5 key dimensions:
  - **Coherence**: Logical structure and flow
  - **Groundedness**: Faithfulness to source material
  - **Relevance**: Focus on important content
  - **Accuracy**: Correctness of information
  - **Completeness**: Comprehensive coverage
- **Dataset Integration**: Uses the arXiv summarization dataset from Hugging Face
- **Visualization**: Generates radar plots for metric comparison
- **Batch Processing**: Evaluates multiple articles efficiently

## Installation

### Prerequisites

- Python 3.13+
- Google API Key for Gemini

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deep_eval_101
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Get your Google API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key
   - Add it to your `.env` file

## Usage

### Running the Jupyter Notebook

The main implementation is in `deep_eval.ipynb`. Open it in Jupyter:

```bash
jupyter notebook deep_eval.ipynb
```

### Key Components

#### 1. Custom LLM Integration

The project implements a custom wrapper for Google Gemini:

```python
class GoogleGenerativeAI(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""
    def __init__(self, model):
        self.model = model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
```

#### 2. Evaluation Metrics

The implementation evaluates abstracts across five dimensions:

- **Coherence**: "Is the abstract logically structured and easy to follow?"
- **Groundedness**: "All claims should be supported by the article"
- **Relevance**: "Focus on most important content, avoid unrelated material"
- **Accuracy**: "Information accurately reflects key details from the article"
- **Completeness**: "Includes all essential information for comprehensive summary"

#### 3. Batch Evaluation

```python
def apply_correctness_metrics_to_dataframe(df, eval_model):
    """Apply evaluation metrics to a dataframe of articles and abstracts"""
    # Processes each row and calculates all metrics
    # Returns dataframe with metric scores and reasons
```

#### 4. Visualization

The notebook includes radar plot visualizations showing metric scores for each evaluated article, making it easy to compare performance across different dimensions.

### Running the Main Script

You can also run the basic script:

```bash
python main.py
```

## Project Structure

```
deep_eval_101/
├── deep_eval.ipynb          # Main implementation notebook
├── main.py                  # Basic script entry point
├── pyproject.toml           # Project dependencies and metadata
├── README.md               # This file
└── uv.lock                 # Lock file for dependencies
```

## Dependencies

- `deepeval>=3.6.7` - Core evaluation framework
- `langchain-google-genai>=2.1.12` - Google Gemini integration
- `datasets>=4.2.0` - Hugging Face datasets
- `matplotlib>=3.10.7` - Visualization
- `python-dotenv>=1.1.1` - Environment variable management
- `ipykernel>=7.0.1` - Jupyter notebook support

## Evaluation Results

The project evaluates academic abstracts and provides:

- **Individual Metric Scores**: 0-1 scale for each evaluation dimension
- **Overall Correctness**: Average of all metric scores
- **Detailed Reasoning**: LLM-generated explanations for each score
- **Visual Comparisons**: Radar plots showing metric distributions
- **Statistical Summaries**: Descriptive statistics across all evaluations

## Use Cases

This implementation is perfect for:

- **Academic Research**: Evaluating summarization quality
- **Content Quality Assessment**: Measuring AI-generated content quality
- **Model Comparison**: Comparing different LLM outputs
- **Educational Purposes**: Learning LLM evaluation techniques
- **Custom Evaluation**: Adapting metrics for specific use cases

## Customization

### Adding New Metrics

To add custom evaluation metrics:

```python
custom_metric = GEval(
    name="Your Metric Name",
    criteria="Your evaluation criteria in natural language",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.INPUT],
    model=eval_model
)
```

### Using Different Datasets

Replace the arXiv dataset with your own:

```python
# Load your custom dataset
your_dataset = load_dataset("your-dataset-name")
df = your_dataset['train'].to_pandas()
```

### Changing the LLM

To use a different LLM provider, implement the `DeepEvalBaseLLM` interface with your preferred model.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Resources

- [DeepEval Documentation](https://deepeval.com/docs/getting-started)
- [Google AI Studio](https://aistudio.google.com/)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [LangChain Google Integration](https://python.langchain.com/docs/integrations/llms/google_vertex_ai_palm)

## Support

For questions and support:
- Check the [DeepEval documentation](https://deepeval.com/docs/getting-started)
- Open an issue in this repository
- Join the DeepEval community discussions
