package ai.eloquent.rephrasing.scripts.idk;

import ai.eloquent.rephrasing.RuleBasedIDKRephraser;
import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.util.ArgumentParser;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class TestRuleBased {

  @ArgumentParser.Option(name = "in", gloss = "The input dataset file. This should be a csv file",
      required = true)
  public static File tsvFile;

  @ArgumentParser.Option(name = "out", gloss = "The input dataset file. This should be a csv file",
      required = false)
  public static File outFile;


  private static List<Sentence> readSentences() {
    List<Sentence> sentences = new ArrayList<>();

    try (CSVReader reader = new CSVReader(new FileReader(tsvFile), '\t')) {
      List<String[]> rows = reader.readAll();
      for (String[] row : rows) {
        if (row.length > 0) {
          sentences.add(new Sentence(row[0]));
        }
      }

    } catch (IOException e) {
      e.printStackTrace();
    }

    return sentences;
  }



  public static void main(String[] args) throws IOException {
    ArgumentParser.fillOptions(TestRuleBased.class, args);
    List<Sentence> sentences = readSentences();
    RuleBasedIDKRephraser rephraser = new RuleBasedIDKRephraser();
    if (outFile == null) {
      for (Sentence sentence : sentences) {
        Optional<String> rephrased = rephraser.rephrased(sentence);
        System.out.println("\n");
        System.out.println(rephrased);
      }
    } else {
      try (CSVWriter writer = new CSVWriter(new FileWriter(outFile), '\t')) {
        int count = 0;
        for (Sentence sentence : sentences) {
          if (count % 100 == 0) {
            System.out.println("" + count + "/" + sentences.size());
          }
          Optional<RuleBasedIDKRephraser.Rephrased> rephrased = rephraser.rephrasedWithRule(sentence);
          String text;
          String rule;
          if (rephrased.isPresent()) {
            text = rephrased.get().toString();
            rule = rephrased.get().rule;
          } else {
            text = "I do not know how to handle '" + sentence.text() + "'";
            rule = "CANONICAL_IDK";
          }
          String[] row = {rule, sentence.text(), text};
          writer.writeNext(row);
          count += 1;
        }
      }
    }
  }
}