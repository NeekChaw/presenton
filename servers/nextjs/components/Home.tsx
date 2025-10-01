"use client";
import { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
// import { toast } from "sonner";
// import { Loader2, Download, CheckCircle } from "lucide-react";
// import { useSelector } from "react-redux";
// import { RootState } from "@/store/store";
// import { handleSaveLLMConfig } from "@/utils/storeHelpers";
// import LLMProviderSelection from "./LLMSelection";
// import {
//   checkIfSelectedOllamaModelIsPulled,
//   pullOllamaModel,
// } from "@/utils/providerUtils";
// import { LLMConfig } from "@/types/llm_config";
// import { trackEvent, MixpanelEvent } from "@/utils/mixpanel";
// import { usePathname } from "next/navigation";
// import {useTranslations} from '@/lib/temporary-translations';

// Button state interface
interface ButtonState {
  isLoading: boolean;
  isDisabled: boolean;
  text: string;
  showProgress: boolean;
  progressPercentage?: number;
  status?: string;
}

export default function Home() {
  const router = useRouter();
  // const pathname = usePathname();
  // const config = useSelector((state: RootState) => state.userConfig);
  // const [llmConfig, setLlmConfig] = useState<LLMConfig>(config.llm_config);

  // const [downloadingModel, setDownloadingModel] = useState<{
  //   name: string;
  //   size: number | null;
  //   downloaded: number | null;
  //   status: string;
  //   done: boolean;
  // } | null>(null);
  // const [showDownloadModal, setShowDownloadModal] = useState<boolean>(false);
  // const [buttonState, setButtonState] = useState<ButtonState>({
  //   isLoading: false,
  //   isDisabled: false,
  //   text: "Save Configuration",
  //   showProgress: false
  // });

  // const canChangeKeys = config.can_change_keys;
  // const downloadProgress = useMemo(() => {
  //   if (downloadingModel && downloadingModel.downloaded !== null && downloadingModel.size !== null) {
  //     return Math.round((downloadingModel.downloaded / downloadingModel.size) * 100);
  //   }
  //   return 0;
  // }, [downloadingModel?.downloaded, downloadingModel?.size]);

  // const handleSaveConfig = async () => {
  //   trackEvent(MixpanelEvent.Home_SaveConfiguration_Button_Clicked, { pathname });
  //   try {
  //     setButtonState(prev => ({
  //       ...prev,
  //       isLoading: true,
  //       isDisabled: true,
  //       text: "Saving Configuration..."
  //     }));
  //     // API: save config
  //     trackEvent(MixpanelEvent.Home_SaveConfiguration_API_Call);
  //     await handleSaveLLMConfig(llmConfig);
  //     if (llmConfig.LLM === "ollama" && llmConfig.OLLAMA_MODEL) {
  //       // API: check model pulled
  //       trackEvent(MixpanelEvent.Home_CheckOllamaModelPulled_API_Call);
  //       const isPulled = await checkIfSelectedOllamaModelIsPulled(llmConfig.OLLAMA_MODEL);
  //       if (!isPulled) {
  //         setShowDownloadModal(true);
  //         // API: download model
  //         trackEvent(MixpanelEvent.Home_DownloadOllamaModel_API_Call);
  //         await handleModelDownload();
  //       }
  //     }
  //     toast.info("Configuration saved successfully");
  //     setButtonState(prev => ({
  //       ...prev,
  //       isLoading: false,
  //       isDisabled: false,
  //       text: "Save Configuration"
  //     }));
  //     // Track navigation from -> to
  //     trackEvent(MixpanelEvent.Navigation, { from: pathname, to: "/upload" });
  //     router.push("/upload");
  //   } catch (error) {
  //     toast.info(error instanceof Error ? error.message : "Failed to save configuration");
  //     setButtonState(prev => ({
  //       ...prev,
  //       isLoading: false,
  //       isDisabled: false,
  //       text: "Save Configuration"
  //     }));
  //   }
  // };

  // const handleModelDownload = async () => {
  //   try {
  //     await pullOllamaModel(llmConfig.OLLAMA_MODEL!, setDownloadingModel);
  //   }
  //   finally {
  //     setDownloadingModel(null);
  //     setShowDownloadModal(false);
  //   }
  // };


  // useEffect(() => {
  //   if (downloadingModel && downloadingModel.downloaded !== null && downloadingModel.size !== null) {
  //     const percentage = Math.round(((downloadingModel.downloaded / downloadingModel.size) * 100));
  //     setButtonState({
  //       isLoading: true,
  //       isDisabled: true,
  //       text: `Downloading Model (${percentage}%)`,
  //       showProgress: true,
  //       progressPercentage: percentage,
  //       status: downloadingModel.status
  //     });
  //   }

  //   if (downloadingModel && downloadingModel.done) {
  //     setTimeout(() => {
  //       setShowDownloadModal(false);
  //       setDownloadingModel(null);
  //       toast.info("Model downloaded successfully!");
  //     }, 2000);
  //   }
  // }, [downloadingModel]);

  // useEffect(() => {
  //   if (!canChangeKeys) {
  //     router.push("/upload");
  //   }
  // }, [canChangeKeys, router]);

  // if (!canChangeKeys) {
  //   return null;
  // }

  return (
    <div className="h-screen bg-gradient-to-b font-instrument_sans from-gray-50 to-white flex flex-col overflow-hidden">
      <main className="flex-1 container mx-auto px-4 max-w-3xl overflow-hidden flex flex-col">
        {/* Branding Header */}
        <div className="text-center mb-2 mt-4 flex-shrink-0">
          <div className="flex items-center justify-center gap-3 mb-2">
            <img src="/Logo.png" alt="Presenton Logo" className="h-12" />
          </div>
          <p className="text-gray-600 text-sm">
            Open-source AI presentation generator
          </p>
        </div>

        {/* Main Configuration Card */}
        <div className="flex-1 overflow-hidden">
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h2 className="text-xl font-semibold mb-4">The Easiest Way to Create Presentations with AI</h2>
            <p className="text-gray-600 mb-6">Just provide a topic, a document, or a description, and let our AI generate a professional presentation for you in minutes.</p>
            <p className="text-sm text-gray-500">Configuration options will be restored soon...</p>
          </div>
        </div>
      </main>

      {/* Download Progress Modal */}
      {/* {showDownloadModal && downloadingModel && (
        ... modal content commented out ...
      )} */}

      {/* Fixed Bottom Button */}
      <div className="flex-shrink-0 bg-white border-t border-gray-200 p-4">
        <div className="container mx-auto max-w-3xl">
          <button
            onClick={() => window.location.href = "/upload"}
            className="w-full font-semibold py-3 px-4 rounded-lg transition-all duration-500 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 focus:ring-4 focus:ring-blue-200 text-white"
          >
            Get Started
          </button>
        </div>
      </div>
    </div>
  );
}
