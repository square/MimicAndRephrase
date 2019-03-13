package ai.eloquent.rephrasing.scripts.idk;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.PropertiesUtils;
import edu.stanford.nlp.util.StringUtils;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class DatasetCleaner {

  public static void main(String[] args) throws IOException {
    Properties props = PropertiesUtils.asProperties(
        "language", "english",
        "annotators", "tokenize,ssplit,truecase",
        "tokenize.class", "PTBTokenizer",
        "tokenize.whitespace", "true",
        "tokenize.language", "en",
        "tokenize.options", "splitHyphenated=true,invertible,ptb3Escaping=true",
        "ssplit.newlineIsSentenceBreak", "two",
        "ner.useSUTime", "false",
        "sutime.markTimeRanges", "true",
        "sutime.includeRange", "true",
        "sutime.rules", "edu/stanford/nlp/models/sutime/defs.sutime.txt,defs.aux.sutime.txt,english.sutime.txt,edu/stanford/nlp/models/sutime/english.holidays.sutime.txt",
        "truecase.overwriteText", "true",
        "truecase.bias", "INIT_UPPER:-2.0,UPPER:-2.0,LOWER:2.5,O:-0.5"  // hand-tuned by Gabor
    );
    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    String dataset = IOUtils.slurpFile("/home/jdieter/eloquent/Datasets/2000dataset.tsv");
    PrintWriter out = new PrintWriter(new FileWriter("/home/jdieter/eloquent/Datasets/2000turkdataset.tsv"));
    for (String line : dataset.split("\n")) {
      String[] fields = line.split("\t");
      if (fields[0].toUpperCase().equals(fields[0])) {
        Annotation ann = new Annotation(fields[0]);
        pipeline.annotate(ann);
        List<String> newTokens = new ArrayList<>();
        List<CoreLabel> questionTokens = ann.get(CoreAnnotations.TokensAnnotation.class);
        for (CoreLabel token : questionTokens) {
          newTokens.add(token.get(CoreAnnotations.TrueCaseTextAnnotation.class));
        }
        fields[0] = StringUtils.join(newTokens, " ");
      }
      if (fields[1].toUpperCase().equals(fields[1])) {
        Annotation ann = new Annotation(fields[1]);
        pipeline.annotate(ann);
        List<String> newTokens = new ArrayList<>();
        List<CoreLabel> questionTokens = ann.get(CoreAnnotations.TokensAnnotation.class);
        for (CoreLabel token : questionTokens) {
          newTokens.add(token.get(CoreAnnotations.TrueCaseTextAnnotation.class));
        }
        fields[1] = StringUtils.join(newTokens, " ");
      }
      out.println(StringUtils.join(fields, "\t"));
    }
    out.close();
  }

}
