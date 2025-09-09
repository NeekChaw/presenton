import React from "react";
import Header from "@/app/(presentation-generator)/dashboard/components/Header";

export const APIKeyWarning: React.FC = () => {
  return (
    <div className="min-h-screen font-roboto bg-gradient-to-br from-slate-50 to-slate-100">
      <Header />
      <div className="flex items-center justify-center aspect-video mx-auto px-6">
        <div className="text-center space-y-4 my-6 bg-white p-10 rounded-lg shadow-lg">
          <h1 className="text-2xl font-bold text-gray-900">
            API Key Required for Template Creation
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            To create AI-powered templates, please configure an API key in the settings page. 
            We support multiple providers:
          </p>
          <div className="grid grid-cols-2 gap-4 mt-6 text-sm">
            <div className="bg-blue-50 p-4 rounded-lg border">
              <h3 className="font-semibold text-blue-800">OpenAI</h3>
              <p className="text-blue-600">Official OpenAI API</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg border">
              <h3 className="font-semibold text-green-800">Custom/OpenRouter</h3>
              <p className="text-green-600">OpenAI-compatible APIs</p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg border">
              <h3 className="font-semibold text-purple-800">Google</h3>
              <p className="text-purple-600">Google Gemini API</p>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg border">
              <h3 className="font-semibold text-orange-800">Anthropic</h3>
              <p className="text-orange-600">Claude API</p>
            </div>
          </div>
          <div className="mt-6">
            <a href="/settings" className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors">
              Configure API Keys
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}; 