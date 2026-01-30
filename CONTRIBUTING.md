# Contributing to Machine Learning Course

Thank you for your interest in contributing! This course aims to make machine learning accessible to everyone.

## 🤝 Ways to Contribute

### 1. Report Bugs
Found an error in code or documentation?
- Open an issue with detailed description
- Include which module and line number
- Describe expected vs actual behavior
- Include error messages if applicable

### 2. Suggest Improvements
Have ideas for better explanations or visualizations?
- Open an issue with your suggestion
- Explain why it would help learners
- Provide examples if possible

### 3. Fix Typos or Errors
Small fixes are welcome!
- Fork the repository
- Make your changes
- Submit a pull request
- Reference the issue number if applicable

### 4. Add New Modules
Want to contribute a new module?
- Open an issue first to discuss
- Follow the module template (see below)
- Include visualizations
- Add YouTube links for complex topics
- Test thoroughly

### 5. Improve Visualizations
Better plots and diagrams always help!
- Maintain the educational style
- Use clear labels and annotations
- Save as high-quality PNG (300 DPI)
- Include source code for regeneration

### 6. Translate Content
Help make this course global!
- Open an issue to coordinate
- Translate module docstrings and comments
- Keep code examples in English
- Maintain formatting

## 📝 Module Template

When creating new modules, follow this structure:

```python
"""
📊 MODULE TITLE - Brief Description
=====================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. First objective
2. Second objective
3. Third objective

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Title"
   https://www.youtube.com/...
   Description

TIME: XX minutes
DIFFICULTY: Beginner/Intermediate/Advanced
PREREQUISITES: Which modules to complete first

KEY CONCEPTS:
------------
- Concept 1
- Concept 2
- Concept 3
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / 'module_name'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

# Module sections with clear headers
# Educational content with examples
# Visualizations with annotations
# Summary at the end
```

## 🎨 Visualization Guidelines

- Use 300 DPI for all saved images
- Include clear labels and titles
- Add annotations to explain key points
- Use color-blind friendly palettes
- Save to appropriate visuals/ subdirectory
- Use descriptive filenames

## 📚 Documentation Standards

- Write clear, beginner-friendly explanations
- Avoid jargon without explanation
- Include real-world examples
- Add comments for complex code
- Link to external resources when helpful
- Use consistent formatting

## 🧪 Testing

Before submitting:
- Run all affected modules
- Verify visualizations generate correctly
- Check for typos and grammar
- Ensure code follows PEP 8 style
- Test on fresh Python environment

## 🔀 Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Commit with clear messages (`git commit -m 'Add XYZ feature'`)
5. Push to your fork (`git push origin feature/YourFeature`)
6. Open a Pull Request with description

## 💬 Code of Conduct

- Be respectful and constructive
- Help beginners feel welcome
- Focus on education, not showing off
- Credit sources and inspirations
- Keep discussions on-topic

## 📧 Questions?

Open an issue or start a discussion if you need clarification!

## 🙏 Thank You!

Every contribution helps make machine learning more accessible. Your effort is appreciated!

---

*Remember: The goal is to help absolute beginners learn ML. Keep explanations simple and clear!*
