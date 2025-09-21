# Contributing to OMR Processing System

Thank you for your interest in contributing to the OMR Processing System! We welcome contributions from the community.

## 🛠️ Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/vegadarsiwork/code4edtech-omreval.git
   cd code4edtech-omreval
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the development servers**
   ```bash
   python flask_backend.py
   streamlit run streamlit_frontend.py --server.port 8502
   ```

## 🎯 How to Contribute

### Reporting Bugs
- Use GitHub Issues to report bugs
- Include steps to reproduce
- Provide system information and error messages
- Add screenshots if applicable

### Suggesting Features
- Open a GitHub Issue with the "enhancement" label
- Describe the feature and its use case
- Explain why it would be valuable

### Code Contributions

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed

3. **Test your changes**
   - Ensure existing functionality works
   - Test with various OMR images
   - Verify both frontend and backend

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Provide clear description
   - Reference any related issues
   - Include screenshots if UI changes

## 📝 Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and concise

### Frontend Code
- Use consistent component structure
- Add comments for complex UI logic
- Ensure responsive design
- Test across different browsers

## 🧪 Testing

### Manual Testing
- Test with various OMR sheet formats
- Verify scoring accuracy
- Check download functionality
- Test error handling

### Areas Needing Tests
- Unit tests for OMR processing functions
- API endpoint testing
- Frontend component testing
- Integration testing

## 📋 Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Changes are tested and working
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files are included
- [ ] Dependencies are properly documented

## 🎯 Priority Areas for Contribution

### High Priority
- Performance optimizations
- Better error handling
- Additional image format support
- Mobile responsiveness improvements

### Medium Priority
- Additional visualization options
- Batch processing features
- API documentation improvements
- Docker containerization

### Future Enhancements
- Machine learning improvements
- Cloud deployment templates
- Mobile app development
- LMS integrations

## 💡 Getting Help

- Check existing issues and discussions
- Ask questions in GitHub Discussions
- Review the README and documentation
- Contact maintainers for guidance

## 🏆 Recognition

Contributors will be recognized in:
- README contributors section
- Release notes for significant contributions
- Special thanks in documentation

Thank you for helping make OMR Processing System better! 🚀
