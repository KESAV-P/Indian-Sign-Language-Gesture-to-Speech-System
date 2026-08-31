import { Ionicons } from "@expo/vector-icons";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as Haptics from "expo-haptics";
import * as Speech from "expo-speech";
import { StatusBar } from "expo-status-bar";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  Dimensions,
  Pressable,
  StyleSheet,
  Text,
  View
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";

import { AnimatedScanFrame } from "./src/components/AnimatedScanFrame";
import { TranslationHistory } from "./src/components/TranslationHistory";
import { WaveformBar } from "./src/components/WaveformBar";
import { createGestureEngine, DetectionSnapshot } from "./src/gestureEngine";

type Facing = "front" | "back";

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get("window");

const ACCENT = "#00D4FF";
const SUCCESS = "#00FF87";
const DARK_BG = "#060A12";
const PANEL_BG = "rgba(8, 14, 26, 0.96)";

const initialSnapshot: DetectionSnapshot = {
  status: "ready",
  caption: "Point camera at an ISL gesture",
  candidate: "Ready",
  confidence: 0,
  handsVisible: false,
  wordHistory: []
};

// ─── Status helpers ────────────────────────────────────────────────────────────

function statusLabel(status: DetectionSnapshot["status"]) {
  switch (status) {
    case "hand_not_visible": return "Searching";
    case "warming_up":       return "Hold Steady";
    case "recognizing":      return "Recognizing";
    case "translated":       return "Translated ✓";
    default:                 return "Live";
  }
}

function statusColor(status: DetectionSnapshot["status"]) {
  switch (status) {
    case "translated":  return SUCCESS;
    case "recognizing": return ACCENT;
    case "warming_up":  return "#FFD166";
    default:            return "rgba(255,255,255,0.55)";
  }
}

// ─── SplashScreen ─────────────────────────────────────────────────────────────

function SplashScreen({ onDone }: { onDone: () => void }) {
  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;
  const dotAnim   = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim,  { toValue: 1, duration: 800, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 800, useNativeDriver: true })
    ]).start();

    // Pulsing dot
    Animated.loop(
      Animated.sequence([
        Animated.timing(dotAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
        Animated.timing(dotAnim, { toValue: 0.3, duration: 600, useNativeDriver: true })
      ])
    ).start();

    const timer = setTimeout(() => {
      Animated.timing(fadeAnim, { toValue: 0, duration: 500, useNativeDriver: true }).start(() =>
        onDone()
      );
    }, 2800);
    return () => clearTimeout(timer);
  }, []);

  return (
    <Animated.View style={[styles.splashRoot, { opacity: fadeAnim }]}>
      <StatusBar style="light" />

      {/* Radial glow behind logo */}
      <View style={styles.splashGlow} />

      <Animated.View
        style={[styles.splashContent, { transform: [{ translateY: slideAnim }] }]}
      >
        {/* Hand emoji with neon ring */}
        <View style={styles.splashIconWrap}>
          <View style={styles.splashIconRing}>
            <Text style={styles.splashEmoji}>🤟</Text>
          </View>
        </View>

        <Text style={styles.splashTitle}>ISL Speak</Text>
        <Text style={styles.splashSub}>Indian Sign Language · AI Translation</Text>

        <View style={styles.splashTagRow}>
          {["Real-time", "On-device", "Demo Mode"].map((tag) => (
            <View key={tag} style={styles.splashTag}>
              <Text style={styles.splashTagText}>{tag}</Text>
            </View>
          ))}
        </View>

        <View style={styles.splashLoadRow}>
          <Animated.View
            style={[
              styles.splashDot,
              { opacity: dotAnim, backgroundColor: ACCENT }
            ]}
          />
          <Text style={styles.splashLoadText}>Starting camera…</Text>
        </View>
      </Animated.View>
    </Animated.View>
  );
}

// ─── Permission screen ─────────────────────────────────────────────────────────

function PermissionScreen({ onRequest }: { onRequest: () => void }) {
  return (
    <View style={styles.permRoot}>
      <StatusBar style="light" />
      <View style={styles.splashGlow} />
      <View style={styles.permCard}>
        <View style={styles.permIconWrap}>
          <Ionicons name="camera" size={38} color="#ffffff" />
        </View>
        <Text style={styles.permTitle}>Camera Access</Text>
        <Text style={styles.permBody}>
          ISL Speak needs the camera to{"\n"}translate gestures live on your device.
        </Text>
        <Pressable
          style={({ pressed }) => [
            styles.permButton,
            { opacity: pressed ? 0.8 : 1 }
          ]}
          onPress={onRequest}
        >
          <Text style={styles.permButtonText}>Allow Camera</Text>
        </Pressable>
      </View>
    </View>
  );
}

// ─── Main Lens Screen ─────────────────────────────────────────────────────────

function LensScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing]             = useState<Facing>("front");
  const [muted, setMuted]               = useState(false);
  const [showSplash, setShowSplash]     = useState(true);
  const [snapshot, setSnapshot]         = useState<DetectionSnapshot>(initialSnapshot);
  const engine    = useMemo(() => createGestureEngine(), []);
  const lastSpoken = useRef("");

  // Caption fade-in animation
  const captionFade = useRef(new Animated.Value(1)).current;
  const prevCaption = useRef("");

  useEffect(() => {
    const timer = setInterval(() => {
      const next = engine.tick();
      setSnapshot(next);

      if (!muted && next.status === "translated" && next.caption && next.caption !== lastSpoken.current) {
        lastSpoken.current = next.caption;
        Speech.speak(next.caption, { rate: 0.92, pitch: 1.0 });
        Haptics.selectionAsync();
      }

      // Animate caption change
      if (next.caption !== prevCaption.current) {
        prevCaption.current = next.caption;
        Animated.sequence([
          Animated.timing(captionFade, { toValue: 0.3, duration: 80, useNativeDriver: true }),
          Animated.timing(captionFade, { toValue: 1, duration: 250, useNativeDriver: true })
        ]).start();
      }
    }, 220);
    return () => clearInterval(timer);
  }, [engine, muted]);

  if (showSplash) {
    return <SplashScreen onDone={() => setShowSplash(false)} />;
  }

  if (!permission) {
    return (
      <View style={[styles.permRoot, { justifyContent: "center", alignItems: "center" }]}>
        <Text style={{ color: "white" }}>Loading…</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return <PermissionScreen onRequest={requestPermission} />;
  }

  const sColor = statusColor(snapshot.status);
  const isTranslating = snapshot.status === "translated" || snapshot.status === "recognizing";
  const confidencePct = Math.round(snapshot.confidence * 100);

  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      <CameraView style={StyleSheet.absoluteFill} facing={facing} />

      {/* Dark vignette gradient overlay */}
      <View style={styles.vignette} pointerEvents="none" />
      <View style={styles.vignetteBottom} pointerEvents="none" />

      <SafeAreaView pointerEvents="box-none" style={StyleSheet.absoluteFill}>

        {/* ── Top Bar ── */}
        <View style={styles.topBar}>
          {/* Brand */}
          <View style={styles.brandBlock}>
            <Text style={styles.brandText}>ISL Speak</Text>
            <View style={styles.statusPill}>
              <Animated.View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor: sColor,
                    shadowColor: sColor,
                    shadowOpacity: 0.9,
                    shadowRadius: 6
                  }
                ]}
              />
              <Text style={[styles.statusLabel, { color: sColor }]}>
                {statusLabel(snapshot.status)}
              </Text>
            </View>
          </View>

          {/* Top-right controls */}
          <View style={styles.topButtons}>
            <TopBtn
              icon={facing === "front" ? "camera-reverse" : "camera-reverse-outline"}
              onPress={() => setFacing((f) => (f === "front" ? "back" : "front"))}
            />
            <TopBtn
              icon={muted ? "volume-mute" : "volume-high"}
              onPress={() => setMuted((m) => !m)}
              active={!muted}
            />
          </View>
        </View>

        {/* ── Scan Frame (center) ── */}
        <View style={styles.scanArea} pointerEvents="none">
          <AnimatedScanFrame
            handsVisible={snapshot.handsVisible}
            translating={isTranslating}
          />
          <Text style={[styles.guideLabel, { color: snapshot.handsVisible ? SUCCESS : "rgba(255,255,255,0.7)" }]}>
            {snapshot.handsVisible ? "✓  Hands detected" : "Keep hands inside frame"}
          </Text>
        </View>

        {/* ── Bottom Translation Dock ── */}
        <View style={styles.dock}>
          <View style={styles.dockBlur}>

            {/* Confidence + Waveform row */}
            <View style={styles.confRow}>
              <View style={styles.confLeft}>
                <Text style={styles.confLabel}>CONFIDENCE</Text>
                <Text style={[styles.confValue, { color: sColor }]}>{confidencePct}%</Text>
              </View>
              <WaveformBar confidence={snapshot.confidence} active={isTranslating} />
            </View>

            {/* Candidate word */}
            <View style={styles.candidateRow}>
              <Text style={styles.candidateLabel}>DETECTED GESTURE</Text>
              <Text style={[styles.candidateWord, { color: ACCENT }]}>
                {snapshot.candidate}
              </Text>
            </View>

            {/* Main caption */}
            <View style={styles.captionBox}>
              <Text style={styles.captionHeader}>LIVE TRANSLATION</Text>
              <Animated.Text
                numberOfLines={3}
                style={[styles.captionText, { opacity: captionFade }]}
              >
                {snapshot.caption}
              </Animated.Text>
            </View>

            {/* Word history chips */}
            {snapshot.wordHistory.length > 0 && (
              <TranslationHistory words={snapshot.wordHistory} />
            )}

            {/* Action buttons */}
            <View style={styles.actions}>
              <ActionBtn
                icon="refresh"
                label="Reset"
                onPress={() => {
                  engine.clear();
                  setSnapshot(initialSnapshot);
                  lastSpoken.current = "";
                }}
              />
              <ActionBtn
                icon="copy-outline"
                label="Copy"
                accent
                onPress={() => {
                  Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                }}
              />
              <ActionBtn
                icon="share-outline"
                label="Share"
                onPress={() => {}}
              />
            </View>

            {/* Demo badge */}
            <View style={styles.demoBadge}>
              <Ionicons name="flask-outline" size={11} color="rgba(255,255,255,0.35)" />
              <Text style={styles.demoText}>  Demo Mode — MediaPipe inference pending</Text>
            </View>
          </View>
        </View>
      </SafeAreaView>
    </View>
  );
}

// ─── Small helper components ───────────────────────────────────────────────────

function TopBtn({
  icon,
  onPress,
  active = false
}: {
  icon: keyof typeof Ionicons.glyphMap;
  onPress: () => void;
  active?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.topBtn,
        active && styles.topBtnActive,
        pressed && { opacity: 0.7 }
      ]}
    >
      <Ionicons name={icon} size={20} color={active ? ACCENT : "#fff"} />
    </Pressable>
  );
}

function ActionBtn({
  icon,
  label,
  onPress,
  accent = false
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
  accent?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.actionBtn,
        accent && styles.actionBtnAccent,
        pressed && { opacity: 0.75 }
      ]}
    >
      <Ionicons name={icon} size={20} color={accent ? DARK_BG : "#ffffff"} />
      <Text style={[styles.actionLabel, accent && { color: DARK_BG }]}>{label}</Text>
    </Pressable>
  );
}

// ─── Root export ───────────────────────────────────────────────────────────────

export default function App() {
  return (
    <SafeAreaProvider>
      <LensScreen />
    </SafeAreaProvider>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  // ── Splash ──
  splashRoot: {
    alignItems: "center",
    backgroundColor: DARK_BG,
    flex: 1,
    justifyContent: "center"
  },
  splashGlow: {
    backgroundColor: "rgba(0, 212, 255, 0.07)",
    borderRadius: 9999,
    height: 400,
    position: "absolute",
    width: 400
  },
  splashContent: {
    alignItems: "center",
    gap: 14,
    paddingHorizontal: 32
  },
  splashIconWrap: {
    marginBottom: 8
  },
  splashIconRing: {
    alignItems: "center",
    borderColor: "rgba(0, 212, 255, 0.5)",
    borderRadius: 50,
    borderWidth: 2,
    height: 100,
    justifyContent: "center",
    shadowColor: ACCENT,
    shadowOpacity: 0.6,
    shadowRadius: 18,
    width: 100
  },
  splashEmoji: {
    fontSize: 52
  },
  splashTitle: {
    color: "#ffffff",
    fontSize: 42,
    fontWeight: "900",
    letterSpacing: -1,
    textAlign: "center"
  },
  splashSub: {
    color: "rgba(255,255,255,0.45)",
    fontSize: 15,
    fontWeight: "600",
    letterSpacing: 0.5,
    textAlign: "center"
  },
  splashTagRow: {
    flexDirection: "row",
    gap: 8,
    marginTop: 4
  },
  splashTag: {
    backgroundColor: "rgba(0, 212, 255, 0.12)",
    borderColor: "rgba(0, 212, 255, 0.3)",
    borderRadius: 20,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 4
  },
  splashTagText: {
    color: ACCENT,
    fontSize: 12,
    fontWeight: "700"
  },
  splashLoadRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 8,
    marginTop: 8
  },
  splashDot: {
    borderRadius: 5,
    height: 8,
    width: 8
  },
  splashLoadText: {
    color: "rgba(255,255,255,0.35)",
    fontSize: 13,
    fontWeight: "600"
  },

  // ── Permission ──
  permRoot: {
    alignItems: "center",
    backgroundColor: DARK_BG,
    flex: 1,
    justifyContent: "center",
    padding: 28
  },
  permCard: {
    alignItems: "center",
    gap: 14,
    maxWidth: 340,
    width: "100%"
  },
  permIconWrap: {
    alignItems: "center",
    backgroundColor: ACCENT,
    borderRadius: 28,
    height: 72,
    justifyContent: "center",
    marginBottom: 8,
    shadowColor: ACCENT,
    shadowOpacity: 0.5,
    shadowRadius: 18,
    width: 72
  },
  permTitle: {
    color: "#ffffff",
    fontSize: 28,
    fontWeight: "900",
    textAlign: "center"
  },
  permBody: {
    color: "rgba(255,255,255,0.5)",
    fontSize: 16,
    lineHeight: 24,
    textAlign: "center"
  },
  permButton: {
    alignItems: "center",
    backgroundColor: ACCENT,
    borderRadius: 18,
    marginTop: 8,
    minHeight: 56,
    justifyContent: "center",
    paddingHorizontal: 28,
    shadowColor: ACCENT,
    shadowOpacity: 0.4,
    shadowRadius: 14,
    width: "100%"
  },
  permButtonText: {
    color: DARK_BG,
    fontSize: 17,
    fontWeight: "900"
  },

  // ── Lens / Camera Screen ──
  root: {
    backgroundColor: DARK_BG,
    flex: 1
  },
  vignette: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "transparent",
    // Top vignette
    borderTopWidth: 120,
    borderTopColor: "rgba(6,10,18,0.75)",
    borderRightWidth: 0,
    borderRightColor: "transparent",
    borderLeftWidth: 0,
    borderLeftColor: "transparent"
  },
  vignetteBottom: {
    bottom: 0,
    height: 260,
    left: 0,
    position: "absolute",
    right: 0,
    // Bottom gradient fade
    backgroundColor: "rgba(6,10,18,0.6)"
  },

  // ── Top bar ──
  topBar: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingTop: 4
  },
  brandBlock: {
    gap: 5
  },
  brandText: {
    color: "#ffffff",
    fontSize: 24,
    fontWeight: "900",
    letterSpacing: -0.5,
    textShadowColor: "rgba(0,0,0,0.6)",
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4
  },
  statusPill: {
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.38)",
    borderRadius: 20,
    flexDirection: "row",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 4
  },
  statusDot: {
    borderRadius: 5,
    height: 8,
    width: 8
  },
  statusLabel: {
    fontSize: 12,
    fontWeight: "800"
  },
  topButtons: {
    flexDirection: "row",
    gap: 8
  },
  topBtn: {
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.42)",
    borderRadius: 24,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
    height: 44,
    justifyContent: "center",
    width: 44
  },
  topBtnActive: {
    backgroundColor: "rgba(0, 212, 255, 0.18)",
    borderColor: ACCENT
  },

  // ── Scan area ──
  scanArea: {
    alignItems: "center",
    flex: 1,
    justifyContent: "center",
    gap: 16,
    paddingBottom: 20
  },
  guideLabel: {
    backgroundColor: "rgba(0,0,0,0.4)",
    borderRadius: 999,
    fontSize: 13,
    fontWeight: "800",
    overflow: "hidden",
    paddingHorizontal: 16,
    paddingVertical: 7
  },

  // ── Bottom dock ──
  dock: {
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    overflow: "hidden"
  },
  dockBlur: {
    backgroundColor: "rgba(6, 10, 18, 0.94)",
    borderColor: "rgba(255,255,255,0.1)",
    borderTopWidth: 1,
    gap: 12,
    paddingBottom: 20,
    paddingHorizontal: 18,
    paddingTop: 18
  },

  // Confidence row
  confRow: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "space-between"
  },
  confLeft: {
    gap: 2
  },
  confLabel: {
    color: "rgba(255,255,255,0.35)",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.5
  },
  confValue: {
    fontSize: 22,
    fontWeight: "900"
  },

  // Candidate
  candidateRow: {
    borderColor: "rgba(255,255,255,0.08)",
    borderRadius: 14,
    borderWidth: 1,
    gap: 3,
    padding: 10
  },
  candidateLabel: {
    color: "rgba(255,255,255,0.35)",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.5
  },
  candidateWord: {
    fontSize: 16,
    fontWeight: "800"
  },

  // Caption
  captionBox: {
    gap: 4
  },
  captionHeader: {
    color: "rgba(255,255,255,0.35)",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.5
  },
  captionText: {
    color: "#ffffff",
    fontSize: 30,
    fontWeight: "900",
    letterSpacing: -0.5,
    lineHeight: 38,
    minHeight: 42,
    textShadowColor: "rgba(0,0,0,0.3)",
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4
  },

  // Actions
  actions: {
    flexDirection: "row",
    gap: 8,
    marginTop: 4
  },
  actionBtn: {
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.1)",
    borderColor: "rgba(255,255,255,0.12)",
    borderRadius: 16,
    borderWidth: 1,
    flex: 1,
    gap: 4,
    justifyContent: "center",
    minHeight: 60
  },
  actionBtnAccent: {
    backgroundColor: ACCENT,
    borderColor: ACCENT
  },
  actionLabel: {
    color: "#ffffff",
    fontSize: 12,
    fontWeight: "800"
  },

  // Demo badge
  demoBadge: {
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    marginTop: -4
  },
  demoText: {
    color: "rgba(255,255,255,0.3)",
    fontSize: 11,
    fontWeight: "600"
  }
});
