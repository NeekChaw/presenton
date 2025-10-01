// Temporary translation hook to replace useTranslations
import { useCallback } from 'react';

// Simple fallback translations - using original en.json content
const fallbackTranslations: Record<string, any> = {
  Home: {
    logoAlt: "Presenton Logo",
    tagline: "Open-source AI presentation generator",
    pageTitle: "Presenton | AI-Powered Presentation Generator",
    title: "The Easiest Way to Create Presentations with AI",
    description: "Just provide a topic, a document, or a description, and let our AI generate a professional presentation for you in minutes.",
    ctaButton: "Get Started",
    githubLink: "View on GitHub"
  },
  Header: {
    createTemplate: "Create Template",
    templates: "Templates"
  },
  LLMSelection: {
    tabs: {
      openai: "OpenAI",
      google: "Google",
      anthropic: "Anthropic",
      ollama: "Ollama",
      custom: "Custom"
    },
    imageProviderLabel: "Select Image Provider",
    searchProviderPlaceholder: "Search provider...",
    noProviderFound: "No provider found.",
    selectImageProvider: "Select image provider",
    apiKeyInfo: "API key for {provider} image generation",
    selectedModelsTitle: "Selected Models",
    selectedModelsInfo: "Using {textModel} for text generation and {imageModel} for images",
    button: {
      saveConfiguration: "Save Configuration",
      savingConfiguration: "Saving Configuration...",
      selectModel: "Please Select a Model",
      enterApiKey: "Please Enter API Key",
      enterOllamaUrl: "Please Enter Ollama URL"
    }
  },
  OpenAIConfig: {
    apiKeyLabel: "OpenAI API Key",
    apiKeyPlaceholder: "Enter your OpenAI API key",
    apiKeyDescription: "Your API key for the OpenAI service.",
    checkModelsButton: "Check for available models",
    checkingModelsButton: "Checking...",
    noModelsFound: "No models found. Please check your API key and URL.",
    modelSelectLabel: "Select OpenAI Model",
    modelSelectPlaceholder: "Select a model",
    searchModelsPlaceholder: "Search models..."
  },
  // Add common defaults for other components
  Dashboard: {
    title: "Dashboard",
    myPresentations: "My Presentations",
    createNew: "Create New",
    recent: "Recent",
  },
  Outline: {
    title: "Review Outline",
    generate: "Generate Presentation",
    edit: "Edit",
    loading: "Generating...",
  },
  Presentation: {
    title: "Presentation",
    export: "Export",
    download: "Download",
    edit: "Edit",
  },
  Upload: {
    title: "Upload Document",
    selectFile: "Select File",
    upload: "Upload",
    processing: "Processing...",
  },
  Configuration: {
    title: "Configuration",
    llm: "Language Model",
    save: "Save",
    cancel: "Cancel",
  },
  CustomConfig: {
    urlLabel: "Custom LLM URL",
    urlPlaceholder: "Enter your custom LLM URL",
    apiKeyLabel: "API Key",
    apiKeyPlaceholder: "Enter your API key (optional)",
    noModelsFound: "No models found. Please check your URL and API key.",
    modelSelectLabel: "Select Model",
    importantNote: "Important: Make sure your custom LLM service is compatible with OpenAI API format.",
    useToolCalls: "Enable Tool Calls",
    toolCallsDescription: "Allow the model to call system functions for structured output",
    disableThinking: "Disable Thinking",
    disableThinkingDescription: "Skip the model's internal reasoning process for faster responses"
  },
  HelpCenter: {
    title: "Help Center",
    ariaLabel: "Help",
    searchPlaceholder: "Search help topics...",
    allCategory: "All",
    noResultsTitle: "No results found",
    noResultsDescription: "Try adjusting your search or browse by category",
    footer: "Need more help? Contact support",
    questions: [
      {
        category: "Getting Started",
        question: "How do I create my first presentation?",
        answer: "Click 'Get Started' from the homepage, upload a document or enter a topic, then follow the guided steps to generate your presentation."
      },
      {
        category: "Configuration",
        question: "How do I configure my AI model?",
        answer: "Go to Settings and select your preferred AI provider (OpenAI, Google, Anthropic, etc.), enter your API key, and choose a model."
      },
      {
        category: "Templates",
        question: "Can I customize presentation templates?",
        answer: "Yes! After generating your outline, you can choose from various templates and customize colors, fonts, and layouts."
      },
      {
        category: "Export",
        question: "What export formats are supported?",
        answer: "You can export your presentations as PowerPoint (.pptx), PDF, or share them online with a generated link."
      }
    ]
  },
  LoadingState: {
    title: "Generating Your Presentation",
    tips: [
      "AI is analyzing your content and creating a structured outline...",
      "Our models work best when you provide clear, specific topics or detailed documents.",
      "The generation process typically takes 30-60 seconds depending on content complexity.",
      "You can customize templates, colors, and layouts after generation is complete.",
      "Pro tip: Upload documents in common formats like PDF, DOCX, or TXT for best results.",
      "The AI considers your content context to create relevant, engaging slides.",
      "Each slide is crafted with appropriate headings, content, and visual suggestions."
    ]
  },
  // Additional namespaces for other components
  AnthropicConfig: {
    apiKeyLabel: "Anthropic API Key",
    apiKeyPlaceholder: "Enter your Anthropic API key",
    modelSelectLabel: "Select Anthropic Model",
  },
  GoogleConfig: {
    apiKeyLabel: "Google API Key",
    apiKeyPlaceholder: "Enter your Google API key",
    modelSelectLabel: "Select Google Model",
  },
  OllamaConfig: {
    urlLabel: "Ollama URL",
    urlPlaceholder: "Enter your Ollama URL",
    modelSelectLabel: "Select Ollama Model",
  },
  OutlinePage: {
    title: "Review Outline",
    subtitle: "Review and edit your presentation outline before generating slides",
  },
  OutlineContent: {
    title: "Presentation Outline",
    edit: "Edit",
    save: "Save",
    cancel: "Cancel",
  },
  LayoutSelection: {
    title: "Choose Template",
    subtitle: "Select a template for your presentation",
  },
  GenerateButton: {
    generate: "Generate Presentation",
    generating: "Generating...",
  },
  EmptyStateView: {
    title: "No outline available",
    description: "Please upload a document or enter a topic to generate an outline.",
  },
  UploadPage: {
    title: "Create New Presentation",
    subtitle: "Upload a document or enter a topic to get started",
  },
  SupportingDoc: {
    title: "Supporting Documents",
    upload: "Upload Document",
    dragDrop: "Drag and drop files here",
  },
  PromptInput: {
    placeholder: "Enter your presentation topic or description...",
    label: "Topic or Description",
  },
  SidePanel: {
    outline: "Outline",
    templates: "Templates",
    settings: "Settings",
  },
  PresentationPage: {
    title: "Presentation",
    export: "Export",
    share: "Share",
  },
  PresentationHeader: {
    title: "Presentation",
    export: "Export",
    share: "Share",
    edit: "Edit",
  },
  ConfigurationSelects: {
    selectTemplate: "Select Template",
    selectLanguage: "Select Language",
    selectTone: "Select Tone",
  },
  DocumentPreview: {
    title: "Document Preview",
    upload: "Upload New",
    continue: "Continue",
  },
  PresentationGrid: {
    myPresentations: "My Presentations",
    createNew: "Create New",
    recent: "Recent",
  },
  PresentationCard: {
    edit: "Edit",
    delete: "Delete",
    share: "Share",
    createdOn: "Created on",
  },
};

export function useTranslations(namespace: string = '') {
  const translate = useCallback((key: string, params?: any) => {
    const keys = key.split('.');
    let value = namespace ? fallbackTranslations[namespace] : fallbackTranslations;

    for (const k of keys) {
      value = value?.[k];
    }

    if (typeof value === 'string') {
      // Simple parameter replacement
      if (params) {
        return value.replace(/\{(\w+)\}/g, (match, param) => params[param] || match);
      }
      return value;
    }

    // Return the raw value for arrays/objects
    if (Array.isArray(value) || typeof value === 'object') {
      return value;
    }

    return key; // Fallback to key if translation not found
  }, [namespace]);

  // Add raw method to return raw values (arrays/objects)
  translate.raw = useCallback((key: string) => {
    const keys = key.split('.');
    let value = namespace ? fallbackTranslations[namespace] : fallbackTranslations;

    for (const k of keys) {
      value = value?.[k];
    }

    return value || [];
  }, [namespace]);

  return translate;
}

// Simple locale hook replacement
export function useLocale() {
  return 'en'; // Default to English
}