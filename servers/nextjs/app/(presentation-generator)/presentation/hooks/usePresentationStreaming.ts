import { useEffect, useRef } from "react";
import { useDispatch } from "react-redux";
import {
  clearPresentationData,
  setPresentationData,
  setStreaming,
} from "@/store/slices/presentationGeneration";
import { jsonrepair } from "jsonrepair";
import { toast } from "sonner";
import { MixpanelEvent, trackEvent } from "@/utils/mixpanel";

export const usePresentationStreaming = (
  presentationId: string,
  stream: string | null,
  setLoading: (loading: boolean) => void,
  setError: (error: boolean) => void,
  fetchUserSlides: () => void
) => {
  const dispatch = useDispatch();
  const previousSlidesLength = useRef(0);

  useEffect(() => {
    let eventSource: EventSource;
    let accumulatedChunks = "";

    const initializeStream = async () => {
      console.log("Initializing presentation stream for ID:", presentationId);
      dispatch(setStreaming(true));
      dispatch(clearPresentationData());

      trackEvent(MixpanelEvent.Presentation_Stream_API_Call);

      // Try direct backend connection to bypass proxy issues
      const streamUrl = `http://localhost:8000/api/v1/ppt/presentation/stream?presentation_id=${presentationId}`;
      console.log("Starting EventSource with direct backend URL:", streamUrl);

      eventSource = new EventSource(streamUrl);

      // Add more detailed event listeners
      eventSource.addEventListener('open', (event) => {
        console.log("EventSource opened:", event);
      });

      eventSource.addEventListener('error', (event) => {
        console.log("EventSource error:", event);
        console.log("ReadyState:", eventSource.readyState);
      });

      eventSource.addEventListener("response", (event) => {
        console.log("EventSource received response event:", event.data);
        const data = JSON.parse(event.data);
        console.log("Parsed response data:", data);

        console.log("Processing event data type:", data.type);
        switch (data.type) {
          case "chunk":
            accumulatedChunks += data.chunk;
            try {
              const repairedJson = jsonrepair(accumulatedChunks);
              const partialData = JSON.parse(repairedJson);

              if (partialData.slides) {
                if (
                  partialData.slides.length !== previousSlidesLength.current &&
                  partialData.slides.length > 0
                ) {
                  dispatch(
                    setPresentationData({
                      ...partialData,
                      slides: partialData.slides,
                    })
                  );
                  previousSlidesLength.current = partialData.slides.length;
                  setLoading(false);
                }
              }
            } catch (error) {
              // JSON isn't complete yet, continue accumulating
            }
            break;

          case "complete":
            try {
              dispatch(setPresentationData(data.presentation));
              dispatch(setStreaming(false));
              setLoading(false);
              eventSource.close();

              // Remove stream parameter from URL
              const newUrl = new URL(window.location.href);
              newUrl.searchParams.delete("stream");
              window.history.replaceState({}, "", newUrl.toString());
            } catch (error) {
              eventSource.close();
              console.error("Error parsing accumulated chunks:", error);
            }
            accumulatedChunks = "";
            break;

          case "closing":
            dispatch(setPresentationData(data.presentation));
            setLoading(false);
            dispatch(setStreaming(false));
            eventSource.close();

            // Remove stream parameter from URL
            const newUrl = new URL(window.location.href);
            newUrl.searchParams.delete("stream");
            window.history.replaceState({}, "", newUrl.toString());
            break;
          case "error":
            eventSource.close();
            toast.error("Error in outline streaming", {
              description:
                data.detail ||
                "Failed to connect to the server. Please try again.",
            });
            setLoading(false);
            dispatch(setStreaming(false));
            setError(true);
            break;
        }
      });

      eventSource.onopen = () => {
        console.log("EventSource connection opened successfully");
      };

      eventSource.onerror = (error) => {
        console.error("EventSource failed:", error);
        console.error("EventSource readyState:", eventSource.readyState);
        setLoading(false);
        dispatch(setStreaming(false));
        setError(true);
        eventSource.close();
      };
    };

    console.log("usePresentationStreaming - stream parameter:", stream);
    console.log("usePresentationStreaming - presentationId:", presentationId);

    if (stream) {
      console.log("Stream detected, initializing stream...");
      initializeStream();
    } else {
      console.log("No stream parameter, fetching user slides...");
      fetchUserSlides();
    }

    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [presentationId, stream, dispatch, setLoading, setError, fetchUserSlides]);
};
