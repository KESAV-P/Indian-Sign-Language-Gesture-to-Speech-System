import React, { useEffect, useRef } from "react";
import { Animated, StyleSheet, View } from "react-native";

type Props = {
  handsVisible: boolean;
  translating: boolean;
};

export function AnimatedScanFrame({ handsVisible, translating }: Props) {
  const glowAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const pulseAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (handsVisible) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(glowAnim, { toValue: 1, duration: 900, useNativeDriver: true }),
          Animated.timing(glowAnim, { toValue: 0.3, duration: 900, useNativeDriver: true })
        ])
      ).start();

      if (translating) {
        Animated.loop(
          Animated.sequence([
            Animated.timing(scaleAnim, { toValue: 1.025, duration: 600, useNativeDriver: true }),
            Animated.timing(scaleAnim, { toValue: 1, duration: 600, useNativeDriver: true })
          ])
        ).start();

        Animated.loop(
          Animated.sequence([
            Animated.timing(pulseAnim, { toValue: 1, duration: 400, useNativeDriver: true }),
            Animated.timing(pulseAnim, { toValue: 0, duration: 400, useNativeDriver: true })
          ])
        ).start();
      } else {
        scaleAnim.stopAnimation();
        pulseAnim.stopAnimation();
        Animated.spring(scaleAnim, { toValue: 1, useNativeDriver: true }).start();
        Animated.spring(pulseAnim, { toValue: 0, useNativeDriver: true }).start();
      }
    } else {
      glowAnim.stopAnimation();
      scaleAnim.stopAnimation();
      Animated.timing(glowAnim, { toValue: 0, duration: 400, useNativeDriver: true }).start();
      Animated.spring(scaleAnim, { toValue: 1, useNativeDriver: true }).start();
    }
  }, [handsVisible, translating]);

  const borderColor = handsVisible
    ? translating
      ? "#00FF87"
      : "#00D4FF"
    : "rgba(255,255,255,0.4)";

  const glowOpacity = glowAnim;

  const cornerSize = 28;
  const cornerThickness = 3;

  const Corner = ({
    top,
    left,
    rotate
  }: {
    top?: boolean;
    left?: boolean;
    rotate: string;
  }) => (
    <View
      style={[
        styles.corner,
        top ? { top: -2 } : { bottom: -2 },
        left ? { left: -2 } : { right: -2 }
      ]}
    >
      <View
        style={[
          styles.cornerH,
          { backgroundColor: borderColor, transform: [{ rotate }] }
        ]}
      />
      <View
        style={[
          styles.cornerV,
          { backgroundColor: borderColor, transform: [{ rotate }] }
        ]}
      />
    </View>
  );

  return (
    <Animated.View
      style={[
        styles.frameOuter,
        {
          transform: [{ scale: scaleAnim }],
          opacity: glowOpacity.interpolate({
            inputRange: [0, 1],
            outputRange: [0.7, 1]
          })
        }
      ]}
    >
      {/* Glow halo */}
      <Animated.View
        style={[
          styles.glowHalo,
          {
            borderColor,
            opacity: glowAnim.interpolate({
              inputRange: [0, 1],
              outputRange: [0, handsVisible ? 0.45 : 0]
            })
          }
        ]}
      />

      {/* Corner brackets */}
      <View style={[styles.frameBorder, { borderColor: "transparent" }]}>
        {/* Top-left */}
        <View style={[styles.cornerBracket, { top: 0, left: 0 }]}>
          <View style={[styles.cH, { backgroundColor: borderColor }]} />
          <View style={[styles.cV, { backgroundColor: borderColor }]} />
        </View>
        {/* Top-right */}
        <View style={[styles.cornerBracket, { top: 0, right: 0 }]}>
          <View style={[styles.cH, { backgroundColor: borderColor, alignSelf: "flex-end" }]} />
          <View style={[styles.cV, { backgroundColor: borderColor, alignSelf: "flex-end" }]} />
        </View>
        {/* Bottom-left */}
        <View style={[styles.cornerBracket, { bottom: 0, left: 0, justifyContent: "flex-end" }]}>
          <View style={[styles.cV, { backgroundColor: borderColor }]} />
          <View style={[styles.cH, { backgroundColor: borderColor }]} />
        </View>
        {/* Bottom-right */}
        <View
          style={[
            styles.cornerBracket,
            { bottom: 0, right: 0, justifyContent: "flex-end", alignItems: "flex-end" }
          ]}
        >
          <View style={[styles.cV, { backgroundColor: borderColor, alignSelf: "flex-end" }]} />
          <View style={[styles.cH, { backgroundColor: borderColor, alignSelf: "flex-end" }]} />
        </View>
      </View>

      {/* Translating pulse ring */}
      {translating && (
        <Animated.View
          style={[
            styles.pulseRing,
            {
              borderColor,
              opacity: pulseAnim.interpolate({
                inputRange: [0, 1],
                outputRange: [0.6, 0]
              }),
              transform: [
                {
                  scale: pulseAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [1, 1.15]
                  })
                }
              ]
            }
          ]}
        />
      )}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  frameOuter: {
    alignItems: "center",
    height: 290,
    justifyContent: "center",
    position: "relative",
    width: "76%"
  },
  glowHalo: {
    borderRadius: 26,
    borderWidth: 12,
    bottom: -10,
    left: -10,
    position: "absolute",
    right: -10,
    top: -10
  },
  frameBorder: {
    borderRadius: 22,
    borderWidth: 0,
    flex: 1,
    position: "relative",
    width: "100%"
  },
  cornerBracket: {
    height: 38,
    position: "absolute",
    width: 38
  },
  cH: {
    borderRadius: 2,
    height: 3,
    width: 38
  },
  cV: {
    borderRadius: 2,
    height: 38,
    position: "absolute",
    width: 3
  },
  pulseRing: {
    borderRadius: 26,
    borderWidth: 2,
    bottom: -4,
    left: -4,
    position: "absolute",
    right: -4,
    top: -4
  },
  corner: {
    height: 32,
    position: "absolute",
    width: 32
  },
  cornerH: {
    borderRadius: 2,
    height: 3,
    position: "absolute",
    width: 32
  },
  cornerV: {
    borderRadius: 2,
    height: 32,
    position: "absolute",
    width: 3
  }
});
