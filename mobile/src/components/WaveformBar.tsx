import React, { useEffect, useRef } from "react";
import { Animated, StyleSheet, View } from "react-native";

type Props = {
  confidence: number; // 0-1
  active: boolean;
};

const BAR_COUNT = 10;

export function WaveformBar({ confidence, active }: Props) {
  const anims = useRef(
    Array.from({ length: BAR_COUNT }, () => new Animated.Value(0.15))
  ).current;

  useEffect(() => {
    if (!active) {
      anims.forEach((anim) => {
        Animated.spring(anim, { toValue: 0.15, useNativeDriver: true }).start();
      });
      return;
    }

    const loops = anims.map((anim, i) => {
      const baseHeight = 0.15 + confidence * 0.7;
      const peak = Math.min(1, baseHeight + 0.25 * Math.sin((i / BAR_COUNT) * Math.PI));
      const duration = 280 + i * 35 + Math.random() * 80;

      return Animated.loop(
        Animated.sequence([
          Animated.timing(anim, {
            toValue: peak,
            duration,
            useNativeDriver: true
          }),
          Animated.timing(anim, {
            toValue: 0.12 + i * 0.015,
            duration,
            useNativeDriver: true
          })
        ])
      );
    });

    loops.forEach((loop) => loop.start());
    return () => loops.forEach((loop) => loop.stop());
  }, [active, confidence]);

  const accentColor = confidence > 0.8 ? "#00FF87" : confidence > 0.5 ? "#00D4FF" : "#7B8EA0";

  return (
    <View style={styles.container}>
      {anims.map((anim, i) => (
        <Animated.View
          key={i}
          style={[
            styles.bar,
            {
              backgroundColor: accentColor,
              opacity: active ? 0.85 + i * 0.01 : 0.25,
              transform: [{ scaleY: anim }]
            }
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "flex-end",
    flexDirection: "row",
    gap: 3,
    height: 28,
    justifyContent: "center"
  },
  bar: {
    borderRadius: 3,
    height: 28,
    width: 4
  }
});
