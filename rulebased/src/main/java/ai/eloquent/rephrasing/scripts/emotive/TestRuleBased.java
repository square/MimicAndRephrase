package ai.eloquent.scripts.rephrasing.emotive;

import ai.eloquent.rephrasing.RuleBasedEmotiveRephraser;
import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.ArgumentParser;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class TestRuleBased {

  @ArgumentParser.Option(name = "in", gloss = "The input dataset file. This should be a tsv file",
      required = true)
  public static File tsvFile;

  @ArgumentParser.Option(name = "out", gloss = "The input dataset file. This should be a tsv file",
      required = false)
  public static File outFile;

  @ArgumentParser.Option(name = "seed", gloss = "Random seed for controlling what is generated",
          required = false)
  public static long seed = 47127099;

  static class SentimentSentence {
    Sentence sentence;
    RuleBasedEmotiveRephraser.Sentiment sentiment;
    boolean isCondensed;

    public SentimentSentence(Sentence sentence, RuleBasedEmotiveRephraser.Sentiment sentiment, boolean isCondensed) {
      this.sentence = sentence;
      this.sentiment = sentiment;
      this.isCondensed = isCondensed;
    }
  }

  private static List<SentimentSentence> readSentences() {
    List<SentimentSentence> sentences = new ArrayList<>();

    try (CSVReader reader = new CSVReader(new FileReader(tsvFile), '\t')) {
      List<String[]> rows = reader.readAll();
      for (String[] row : rows) {
        if (row.length > 0) {
          RuleBasedEmotiveRephraser.Sentiment sentiment = "pos".equals(row[2])?
                  RuleBasedEmotiveRephraser.Sentiment.Positive : RuleBasedEmotiveRephraser.Sentiment.Negative;
          boolean isCondensed = "condensed".equals(row[3]);
          sentences.add(
                  new SentimentSentence(new Sentence(row[0]), sentiment, isCondensed));
        }
      }

    } catch (IOException e) {
      e.printStackTrace();
    }

    return sentences;
  }



  public static void main(String[] args) throws IOException {
    ArgumentParser.fillOptions(TestRuleBased.class, args);
    List<SentimentSentence> sentences = readSentences();
    RuleBasedEmotiveRephraser rephraser = new RuleBasedEmotiveRephraser();
    rephraser.setSeed(seed);
    if (outFile == null) {
      for (SentimentSentence sentence : sentences) {
        Optional<String> rephrased = rephraser.rephrased(sentence.sentence, sentence.sentiment, sentence.isCondensed);
        System.out.println("\n");
        System.out.println(rephrased);
      }
    } else {
      try (CSVWriter writer = new CSVWriter(new FileWriter(outFile), '\t')) {
        int count = 0;
        for (SentimentSentence sentence : sentences) {
          if (count % 100 == 0) {
            System.out.println("" + count + "/" + sentences.size());
          }
          Optional<String> rephrased = rephraser.rephrased(sentence.sentence, sentence.sentiment, sentence.isCondensed);
          String toAdd;
          if (rephrased.isPresent()) {
            toAdd = rephrased.get();
          } else {
            toAdd = rephraser.rephrasedGeneric(sentence.sentiment);
          }
          String sentiment = sentence.sentiment.name().toLowerCase().substring(0,3);
          String[] row = {sentence.sentence.text(), toAdd, sentiment, sentence.isCondensed? "condensed" : "full"};
          writer.writeNext(row);
          count += 1;
        }
      }
    }
  }
}