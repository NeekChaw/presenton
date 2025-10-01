import { useState, useCallback } from "react";
import { useTranslations } from "@/lib/temporary-translations";
import { useDispatch } from "react-redux";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { clearPresentationData } from "@/store/slices/presentationGeneration";
import { PresentationGenerationApi } from "../../services/api/presentation-generation";
import { LayoutGroup, LoadingState, TABS } from "../types/index";
import { MixpanelEvent, trackEvent } from "@/utils/mixpanel";

const DEFAULT_LOADING_STATE: LoadingState = {
  message: "",
  isLoading: false,
  showProgress: false,
  duration: 0,
};

export const usePresentationGeneration = (
  presentationId: string | null,
  outlines: { content: string }[] | null,
  selectedLayoutGroup: LayoutGroup | null,
  setActiveTab: (tab: string) => void
) => {
  const t = useTranslations('PresentationGeneration');
  const dispatch = useDispatch();
  const router = useRouter();
  const [loadingState, setLoadingState] = useState<LoadingState>(DEFAULT_LOADING_STATE);

  const validateInputs = useCallback(() => {
    if (!outlines || outlines.length === 0) {
      toast.error(t('noOutlinesError'), {
        description: t('noOutlinesDescription'),
      });
      return false;
    }

    if (!selectedLayoutGroup) {
      toast.error(t('selectLayoutError'), {
        description: t('selectLayoutDescription'),
      });
      return false;
    }
    if (!selectedLayoutGroup.slides.length) {
      toast.error(t('noSlideSchemaError'), {
        description: t('noSlideSchemaDescription'),
      });
      return false;
    }

    return true;
  }, [outlines, selectedLayoutGroup, t]);

  const prepareLayoutData = useCallback(() => {
    if (!selectedLayoutGroup) return null;
    return {
      name: selectedLayoutGroup.name,
      ordered: selectedLayoutGroup.ordered,
      slides: selectedLayoutGroup.slides
    };
  }, [selectedLayoutGroup]);

  const handleSubmit = useCallback(async () => {
    console.log("handleSubmit called with:", {
      selectedLayoutGroup,
      presentationId,
      outlines: outlines?.length || 0
    });

    if (!selectedLayoutGroup) {
      console.log("No layout selected, switching to layouts tab");
      setActiveTab(TABS.LAYOUTS);
      return;
    }
    if (!validateInputs()) {
      console.log("Validation failed");
      return;
    }

    console.log("Starting presentation generation...");

    setLoadingState({
      message: t('loadingMessage'),
      isLoading: true,
      showProgress: true,
      duration: 30,
    });

    try {
      const layoutData = prepareLayoutData();
      console.log("Layout data prepared:", layoutData);

      if (!layoutData) {
        console.log("No layout data, returning");
        return;
      }

      trackEvent(MixpanelEvent.Presentation_Prepare_API_Call);

      console.log("Calling PresentationGenerationApi.presentationPrepare...");
      const response = await PresentationGenerationApi.presentationPrepare({
        presentation_id: presentationId,
        outlines: outlines,
        layout: layoutData,
      });

      console.log("API response received:", response);

      if (response) {
        console.log("Success! Navigating to presentation page...");
        dispatch(clearPresentationData());
        router.replace(`/presentation?id=${presentationId}&stream=true`);
      }
    } catch (error: any) {
      console.error('Error In Presentation Generation(prepare).', error);
      console.error('Full error object:', error);
      toast.error(t('generationError'), {
        description: error.message || t('generationErrorDescription'),
      });
    } finally {
      console.log("Resetting loading state");
      setLoadingState(DEFAULT_LOADING_STATE);
    }
  }, [validateInputs, prepareLayoutData, presentationId, outlines, dispatch, router, t]);

  return { loadingState, handleSubmit };
}; 