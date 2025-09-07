import { useState, useCallback } from "react";
import { useTranslations } from "next-intl";
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
    if (!selectedLayoutGroup) {
      setActiveTab(TABS.LAYOUTS);
      return;
    }
    if (!validateInputs()) return;



    setLoadingState({
      message: t('loadingMessage'),
      isLoading: true,
      showProgress: true,
      duration: 30,
    });

    try {
      const layoutData = prepareLayoutData();

      if (!layoutData) return;
      trackEvent(MixpanelEvent.Presentation_Prepare_API_Call);
      const response = await PresentationGenerationApi.presentationPrepare({
        presentation_id: presentationId,
        outlines: outlines,
        layout: layoutData,
      });

      if (response) {
        dispatch(clearPresentationData());
        router.replace(`/presentation?id=${presentationId}&stream=true`);
      }
    } catch (error: any) {
      console.error('Error In Presentation Generation(prepare).', error);
      toast.error(t('generationError'), {
        description: error.message || t('generationErrorDescription'),
      });
    } finally {
      setLoadingState(DEFAULT_LOADING_STATE);
    }
  }, [validateInputs, prepareLayoutData, presentationId, outlines, dispatch, router, t]);

  return { loadingState, handleSubmit };
}; 