/**
 * Gesture Engine — Real MediaPipe + PyTorch Inference via Backend API
 *
 * Replaces the previous simulation ticker.
 * The engine sends base64-encoded camera frames to the FastAPI backend's
 * POST /predict_frame endpoint, which runs MediaPipe landmark extraction
 * and LSTM inference server-side, then returns a DetectionSnapshot.
 *
 * Architecture:
 *   Mobile camera → (base64 JPEG) → /predict_frame → FramePredictResponse
 *                                                        ↓
 *                                               DetectionSnapshot (same
 *                                               interface as before, so
 *                                               App.tsx needs minimal changes)
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export type DetectionStatus =
  | "ready"
  | "hand_not_visible"
  | "warming_up"
  | "recognizing"
  | "translated";

export type DetectionSnapshot = {
  status: DetectionStatus;
  caption: string;
  candidate: string;
  confidence: number;
  handsVisible: boolean;
  wordHistory: string[];
};

// ── Backend URL configuration ─────────────────────────────────────────────────
//
// When running in Expo Go on a PHYSICAL device, set this to your Mac's
// LAN IP address (e.g. "http://192.168.1.100:8000").
// The iOS Simulator and Android Emulator can use "http://localhost:8000".
//
// IMPORTANT: Change this to your Mac's LAN IP when testing on a real device.
// Find it with: `ipconfig getifaddr en0`  (macOS WiFi IP)
//
const BACKEND_URL = "http://localhost:8000";

// Session ID for server-side window state (one per app session)
const SESSION_ID = "mobile_default";

// ── API response shape (must match backend's FramePredictResponse) ────────────

type FramePredictResponse = {
  label: string;
  confidence: number;
  status: string;
  hands_visible: boolean;
  sentence: string;
  sequence_fill: number;
  candidate: string;
};

// ── GestureApiClient ──────────────────────────────────────────────────────────

export type GestureEngine = {
  /** Send a base64-encoded JPEG frame to the backend and return a DetectionSnapshot. */
  processFrame: (imageBase64: string) => Promise<DetectionSnapshot>;
  /** Reset server-side window state and clear local word history. */
  clear: () => Promise<DetectionSnapshot>;
  /** Return current snapshot without sending a new frame (for polling fallback). */
  getSnapshot: () => DetectionSnapshot;
};

const initialSnapshot: DetectionSnapshot = {
  status: "ready",
  caption: "Point camera at an ISL gesture",
  candidate: "Ready",
  confidence: 0,
  handsVisible: false,
  wordHistory: [],
};

export function createGestureEngine(): GestureEngine {
  // Local state mirrored from server responses
  let lastSnapshot: DetectionSnapshot = { ...initialSnapshot };
  let wordHistory: string[] = [];
  let lastAcceptedWord = "";

  /**
   * Map a FramePredictResponse to a DetectionSnapshot.
   * - `caption`     = accumulated sentence (from sentence_buffer)
   * - `candidate`   = current candidate word from model
   * - `wordHistory` = last 5 accepted words (for word chips in UI)
   */
  function mapResponse(resp: FramePredictResponse): DetectionSnapshot {
    // Update word history when a new word is translated
    if (
      resp.status === "translated" &&
      resp.label &&
      resp.label !== lastAcceptedWord &&
      resp.label !== "Ready"
    ) {
      lastAcceptedWord = resp.label;
      wordHistory = [...wordHistory.slice(-4), resp.label];
    }

    // Map server status string to DetectionStatus type
    const status = mapStatus(resp.status);

    // Caption = the full sentence assembled by the server's SentenceBuffer
    const caption =
      resp.sentence ||
      captionForStatus(status, resp.candidate, resp.sequence_fill);

    return {
      status,
      caption,
      candidate: resp.candidate || resp.label || "—",
      confidence: resp.confidence,
      handsVisible: resp.hands_visible,
      wordHistory: [...wordHistory],
    };
  }

  function mapStatus(s: string): DetectionStatus {
    switch (s) {
      case "translated":       return "translated";
      case "recognizing":      return "recognizing";
      case "warming_up":       return "warming_up";
      case "hand_not_visible": return "hand_not_visible";
      default:                 return "ready";
    }
  }

  function captionForStatus(
    status: DetectionStatus,
    candidate: string,
    fill: number,
  ): string {
    switch (status) {
      case "hand_not_visible": return "Move hands into the frame";
      case "warming_up":       return `Collecting frames… ${Math.round(fill * 100)}%`;
      case "recognizing":      return `Recognizing "${candidate}"…`;
      case "translated":       return candidate;
      default:                 return "Point camera at an ISL gesture";
    }
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  async function processFrame(imageBase64: string): Promise<DetectionSnapshot> {
    try {
      const resp = await fetch(`${BACKEND_URL}/predict_frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_b64: imageBase64,
          session_id: SESSION_ID,
        }),
        // 2-second timeout so a slow backend doesn't freeze the UI
        signal: AbortSignal.timeout(2000),
      });

      if (!resp.ok) {
        const errText = await resp.text().catch(() => "");
        console.warn(`[GestureEngine] /predict_frame ${resp.status}: ${errText}`);
        // Return last known snapshot on API error
        return lastSnapshot;
      }

      const data: FramePredictResponse = await resp.json();
      lastSnapshot = mapResponse(data);
      return lastSnapshot;
    } catch (err) {
      // Network error or timeout — surface as "hand not visible" to avoid frozen UI
      if (lastSnapshot.status !== "hand_not_visible") {
        console.warn("[GestureEngine] Network error:", err);
      }
      // Only downgrade status if we haven't recently translated something
      if (lastSnapshot.status === "ready" || lastSnapshot.status === "hand_not_visible") {
        lastSnapshot = {
          ...lastSnapshot,
          status: "hand_not_visible",
          handsVisible: false,
          confidence: 0,
          candidate: "Connecting to backend…",
        };
      }
      return lastSnapshot;
    }
  }

  async function clear(): Promise<DetectionSnapshot> {
    // Reset server-side window state
    try {
      await fetch(`${BACKEND_URL}/session/reset?session_id=${SESSION_ID}`, {
        method: "POST",
        signal: AbortSignal.timeout(2000),
      });
    } catch (e) {
      console.warn("[GestureEngine] Reset request failed:", e);
    }

    // Reset local state
    wordHistory = [];
    lastAcceptedWord = "";
    lastSnapshot = { ...initialSnapshot };
    return lastSnapshot;
  }

  function getSnapshot(): DetectionSnapshot {
    return lastSnapshot;
  }

  return { processFrame, clear, getSnapshot };
}
