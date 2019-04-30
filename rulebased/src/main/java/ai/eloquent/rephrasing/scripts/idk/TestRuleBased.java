package ai.eloquent.rephrasing.scripts.idk;

import ai.eloquent.rephrasing.RuleBasedIDKRephraser;
import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.ArgumentParser;

import java.io.*;
import java.text.DateFormat;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Optional;

public class TestRuleBased {

  @ArgumentParser.Option(name = "in", gloss = "The input dataset file. This should be a csv file",
      required = true)
  public static File tsvFile;

  @ArgumentParser.Option(name = "out", gloss = "The input dataset file. This should be a csv file",
      required = false)
  public static File outFile;


  private static List<String> readSentences() {
    List<String> sentences = new ArrayList<>();

    try (CSVReader reader = new CSVReader(new FileReader(tsvFile), '\t')) {
      List<String[]> rows = reader.readAll();
      for (String[] row : rows) {
        if (row.length > 0) {
          sentences.add(row[0]);
        }
      }

    } catch (IOException e) {
      e.printStackTrace();
    }

    return sentences;
  }


  public static void main(String[] args) throws IOException {
    DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
    System.out.println(dateFormat.format(new Date()));
    ArgumentParser.fillOptions(TestRuleBased.class, args);
    List<String> sentences = readSentences();
    RuleBasedIDKRephraser rephraser = new RuleBasedIDKRephraser();
    if (outFile == null) {
      for (String sentence : sentences) {
        Optional<String> rephrased = rephraser.rephrased(sentence);
        System.out.println("\n");
        System.out.println(rephrased);
      }
    } else {
      try (CSVWriter writer = new CSVWriter(new FileWriter(outFile), '\t')) {
        int count = 0;
        int unhandled = 0;
        IntCounter<String> ruleCounter = new IntCounter<>();
        for (String sentence : sentences) {
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
            text = "I do not know how to handle '" + sentence + "'";
            rule = "CANONICAL_IDK";
            unhandled++;
          }
          ruleCounter.incrementCount(rule);
          String[] row = {sentence, text, rule};
          writer.writeNext(row);
          count += 1;
        }
        System.out.println("Unhandled: " + unhandled + "/" + count + "=" + ((double) unhandled/count));
        System.out.println(ruleCounter.toString(NumberFormat.getInstance()));
      }
    }
    System.out.println(dateFormat.format(new Date()));
  }
}