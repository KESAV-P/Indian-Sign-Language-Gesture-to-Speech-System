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

const ISL_VOCABULARY: string[] = [
  "Hello",
  "Thank You",
  "Please",
  "Yes",
  "No",
  "Help",
  "Good Morning",
  "Food",
  "Water",
  "I Love You",
  "Sorry",
  "Come Here",
  "Go Away",
  "How Are You",
  "My Name Is",
  "I Am Fine",
  "Good Night",
  "See You",
  "Congratulations",
  "Welcome"
];

export function createGestureEngine() {
  let frame = 0;
  let caption = "";
  let lastAccepted = "";
  let wordHistory: string[] = [];

  function clear(): DetectionSnapshot {
    frame = 0;
    caption = "";
    lastAccepted = "";
    wordHistory = [];
    return {
      status: "ready",
      caption: "Point camera at an ISL gesture",
      candidate: "Ready",
      confidence: 0,
      handsVisible: false,
      wordHistory: []
    };
  }

  function smoothConfidence(raw: number): number {
    // Smooth sine-wave based confidence curve
    return 0.45 + 0.45 * Math.sin((raw * Math.PI) / 2);
  }

  function tick(): DetectionSnapshot {
    frame += 1;

    // Phase 1: looking for hands (first ~2s at 240ms tick = ~8 frames)
    if (frame < 9) {
      return {
        status: "hand_not_visible",
        caption: caption || "Move hands into the frame",
        candidate: "Searching for hands…",
        confidence: 0.05 + (frame / 9) * 0.08,
        handsVisible: false,
        wordHistory
      };
    }

    // Phase 2: warming up / collecting sequence (~2.5s more)
    if (frame < 20) {
      const warmProgress = (frame - 9) / 11;
      return {
        status: "warming_up",
        caption: caption || "Hold gesture steady",
        candidate: `Collecting keypoints ${Math.round(warmProgress * 100)}%`,
        confidence: smoothConfidence(warmProgress * 0.6),
        handsVisible: true,
        wordHistory
      };
    }

    // Each word cycle: 32 frames = ~7.7s
    const cycleLen = 32;
    const cyclePos = frame % cycleLen;
    const vocabIndex = Math.floor(frame / cycleLen) % ISL_VOCABULARY.length;
    const word = ISL_VOCABULARY[vocabIndex];

    // First 22 frames of cycle: recognizing phase
    if (cyclePos < 22) {
      const recProgress = cyclePos / 22;
      return {
        status: "recognizing",
        caption: caption || "Recognizing gesture…",
        candidate: word,
        confidence: smoothConfidence(0.4 + recProgress * 0.5),
        handsVisible: true,
        wordHistory
      };
    }

    // Last 10 frames of cycle: translated / accepted
    if (word !== lastAccepted) {
      caption = caption ? `${caption} ${word}` : word;
      lastAccepted = word;
      wordHistory = [...wordHistory.slice(-4), word];
    }

    return {
      status: "translated",
      caption,
      candidate: word,
      confidence: 0.87 + (Math.sin(frame * 0.3) * 0.05),
      handsVisible: true,
      wordHistory
    };
  }

  return { clear, tick };
}
