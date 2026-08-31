import React from "react";
import { ScrollView, StyleSheet, Text, View } from "react-native";

type Props = {
  words: string[];
};

export function TranslationHistory({ words }: Props) {
  if (words.length === 0) return null;

  return (
    <View style={styles.container}>
      <Text style={styles.label}>HISTORY</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
      >
        {words.map((word, i) => {
          const isLatest = i === words.length - 1;
          return (
            <View
              key={`${word}-${i}`}
              style={[
                styles.chip,
                isLatest ? styles.chipActive : styles.chipInactive
              ]}
            >
              <Text
                style={[
                  styles.chipText,
                  isLatest ? styles.chipTextActive : styles.chipTextInactive
                ]}
              >
                {word}
              </Text>
            </View>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: 6
  },
  label: {
    color: "rgba(255,255,255,0.35)",
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 1.5,
    paddingLeft: 2
  },
  scroll: {
    flexDirection: "row",
    gap: 8,
    paddingRight: 4
  },
  chip: {
    borderRadius: 20,
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 5
  },
  chipActive: {
    backgroundColor: "rgba(0, 212, 255, 0.18)",
    borderColor: "#00D4FF"
  },
  chipInactive: {
    backgroundColor: "rgba(255,255,255,0.06)",
    borderColor: "rgba(255,255,255,0.15)"
  },
  chipText: {
    fontSize: 13,
    fontWeight: "700"
  },
  chipTextActive: {
    color: "#00D4FF"
  },
  chipTextInactive: {
    color: "rgba(255,255,255,0.55)"
  }
});
