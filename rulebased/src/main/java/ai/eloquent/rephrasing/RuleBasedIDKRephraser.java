package ai.eloquent.rephrasing;

import edu.stanford.nlp.coref.data.WordLists;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.naturalli.VerbTense;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class RuleBasedIDKRephraser extends RuleBasedRephraser {

  public Optional<String> rephrased(Sentence request) {
    Optional<Rephrased> rephrased = rephrasedWithRule(request);
    return rephrased.map(x -> x.toString());
  }

  public Optional<String> rephrased(String request) {
    Optional<Rephrased> rephrased = rephrasedWithRule(request);
    return rephrased.map(x -> x.toString());
  }

  public Optional<Rephrased> rephrasedWithRule(String request) {
    Annotation ann = new Annotation(request);
    TOKENIZE_WS_SSPLIT.get().annotate(ann);
    Document doc = new Document(ann);
    return selectOneRephrasedWithRule(doc.sentences());
  }

  public Optional<Rephrased> selectOneRephrasedWithRule(List<Sentence> sentences) {
    Optional<Rephrased> best = Optional.empty();
    for (Sentence sentence : sentences) {
      Optional<Rephrased> rephrased = rephrasedWithRule(sentence);
      if (!best.isPresent()) {
        best = rephrased;
      }
    }
    return best;
  }


  public Optional<Rephrased> rephrasedWithRule(Sentence request) {
    //Preprocess the sentence by switching the point of view of pronouns i.e. I to you
    request.lemmas(); // Make sure we have lemmas
    Sentence simplifiedRequest = simplify(request);
    Optional<Rephrased> rephrased = rephraseQuestion(simplifiedRequest, request);
    if (rephrased.isPresent()) {
      //Make sure not upper cased
      rephrased.get().sentence = rephrased
              .map(x -> removeQuestionMark(x.sentence))
              .map(x -> adjustCapitalization(x))
              .map(x -> replacePronouns(x)).get();

      //Let's ignore the condensed flag
      return rephrased;
    } else {
      List<Sentence> clauses = getClauses(request);
      if (clauses.size() > 1) {
        return selectOneRephrasedWithRule(clauses);
      } else {
        return Optional.empty();
      }
    }
  }

  public static class Rephrased {
    public Sentence sentence;  // Rephrased sentence
    public String prefix;      // IDK prefix to apply
    public String rule;        // Rule that was applied

    public Rephrased(String prefix, Sentence sentence, String rule) {
      this.sentence = sentence;
      this.prefix = prefix;
      this.rule = rule;
    }

    public String toString() {
      return prefix + " " + sentence;
    }
  }

  private Sentence removeQuestionMark(Sentence sentence) {
    SemanticGraph depGraph = sentence.dependencyGraph();
    SemgrexPattern qPuncPattern = SemgrexPattern.compile("{word:/\\?/}");
    SemgrexMatcher qMatcher = qPuncPattern.matcher(depGraph);
    if (qMatcher.find()) {
      depGraph.removeVertex(qMatcher.getMatch());
    }
    return new Sentence(depGraph.toRecoveredSentenceString());
  }


  // Patterns at beginning of sentences indicating need or want: I need, I want, I would like
  TokenSequencePattern needPattern = TokenSequencePattern.compile("[lemma:I] [lemma:would]? [lemma:/need|want|like/] [lemma:to]?");
//  TokenSequencePattern needPattern = TokenSequencePattern.compile("/I/ /need/");
  private Sentence simplify(Sentence sentence) {
    List<String> words = sentence.words();
    if (words.size() > 1 && words.get(0).equalsIgnoreCase("please")) {
      // Drop please
      sentence = new Sentence(words.subList(1, words.size()));
      words = sentence.words();
    }

    // Remove "I need"
    List<Pair<Integer,Integer>> matchedOffsets = sentence.find(needPattern, m -> Pair.makePair(m.start(), m.end()));
    matchedOffsets = matchedOffsets.stream().filter(x -> x.first == 0).collect(Collectors.toList());
    if (matchedOffsets.size() > 0) {
      List<String> newWords = new ArrayList<>();
      int start = 0;
      for (Pair<Integer,Integer> p : matchedOffsets) {
        newWords.addAll(words.subList(0, p.first));
        start = p.second;
      }
      if (start < words.size()) {
        newWords.addAll(words.subList(start, words.size()));
      }
      if (newWords.size() > 0) {
        sentence = new Sentence(newWords);
      }
    }
    sentence = removeAuxilliaryDo(sentence);
    return sentence;
  }

  private IndexedWord findSubj(SemanticGraph graph, IndexedWord verb) {
    List<SemanticGraphEdge> edges = graph.getOutEdgesSorted(verb);
    for (SemanticGraphEdge e : edges) {
      if (e.getRelation().getShortName().equals("nsubj")) {
        return e.getTarget();
      }
    }
    return null;
  }

  private boolean isPlural(IndexedWord w) {
    if (w.tag().equals("PRP")) {
      return WordLists.pluralPronounsEn.contains(w.lemma().toLowerCase());
    } else if (w.tag().endsWith("S")) {
      return true;
    } else {
      return false;
    }
  }

  private int getPerson(IndexedWord w) {
    if (w.tag().equals("PRP")) {
      if (WordLists.firstPersonPronounsEn.contains(w.lemma().toLowerCase())) {
        return 1;
      } else if (WordLists.secondPersonPronounsEn.contains(w.lemma().toLowerCase())) {
        return 2;
      } else {
        return 3;
      }
    } else {
      return 3;
    }
  }

  private Sentence removeAuxilliaryDo(Sentence sentence) {
    SemanticGraph dependencyGraph = sentence.dependencyGraph();
    // Simplify sentences like "Why do you sell socks?" to "Why you sell socks"
    // Simplify sentences like "Why did you sell socks?" to "Why you sold socks"
    // Simplify sentences like "Do you sell socks?" to "You sell socks?"
    SemgrexPattern auxPattern = SemgrexPattern.compile("{pos:/V.*/}=v1 >aux {pos:/V.*/}=v2");
    SemgrexMatcher auxPatternMatcher = auxPattern.matcher(dependencyGraph);
    if (auxPatternMatcher.find()) {
      IndexedWord v1 = auxPatternMatcher.getNode("v1");
      IndexedWord v2 = auxPatternMatcher.getNode("v2");
      if (v2.lemma() != null && v2.lemma().equalsIgnoreCase("do")) {
        IndexedWord subj = findSubj(dependencyGraph, v1);
        boolean isPlural = (subj != null)? isPlural(subj) : false;
        int person = (subj != null)? getPerson(subj) : 3;
        if (v2.tag().equals("VBD")) {
          // Past tense
          VerbTense t = VerbTense.of(true, isPlural, false, person);
          v1.setWord(t.conjugateEnglish(v1.backingLabel()));
        } else if (v2.tag().equals("VBZ")) {
          VerbTense t = VerbTense.of(false, isPlural, false, person);
          v1.setWord(t.conjugateEnglish(v1.backingLabel()));
        }
        dependencyGraph.removeVertex(v2);
        sentence = new Sentence(dependencyGraph.toRecoveredSentenceString());
      }
    }
    return sentence;
  }

  private Optional<Rephrased> rephraseQuestion(Sentence sentence, Sentence originalSentence) {
    //Get graph for semgrex searches
    SemanticGraph dependencyGraph = sentence.dependencyGraph();

    //Check for imperative sentence (no subject, no W pos, no question mark)
    SemgrexPattern subjPattern = SemgrexPattern.compile("{} >nsubj {}");
    SemgrexPattern wPattern = SemgrexPattern.compile("{pos:/W.*/}");
    SemgrexPattern qPuncPattern = SemgrexPattern.compile("{word:/\\?/}");
    // Handles sentences like "Help me"
    if (!subjPattern.matcher(dependencyGraph).find() && !wPattern.matcher(dependencyGraph).find() && !qPuncPattern.matcher(dependencyGraph).find()) {
      return Optional.of(new Rephrased("I do not know how to", sentence, "IMPERATIVE"));
    }

    //Check for easy question to switch around the phrasing
    SemanticGraph copyDepGraph = new SemanticGraph(dependencyGraph);

    //System.out.println(copyDepGraph);
    // Find the verb which is a parent of a W-word and a parent of some (nominal) subject (that is not the W-word).
    SemgrexPattern qPattern1 = SemgrexPattern.compile("{pos:/V.*/} > {pos:/W.*/}=w >/nsubj.*/ ({}=s !== {}=w)");
    // A W-word t's the parent of a clause containing a verb and also of a subject
    SemgrexPattern qPattern2 = SemgrexPattern.compile("{pos:/W.*/} > {pos:/V.*/}=v >/nsubj.*/ {}=s");
    // Find the verb which is a parent of a noun-phrase containing a W-word and a parent of some (nominal) subject (that is not the W-word).
    SemgrexPattern qPattern3 = SemgrexPattern.compile("{pos:/V.*/} > ({}=w >> {pos:/W.*/}) >/nsubj.*/ ({}=s !== {}=w)");
    // A W-word t's the parent of a clause containing a (verb and a subject)
    SemgrexPattern qPattern4 = SemgrexPattern.compile("{pos:/W.*/} > ({pos:/V.*/}=v >/nsubj.*/ {}=s)");
    SemgrexMatcher qMatcher1 = qPattern1.matcher(copyDepGraph);
    SemgrexMatcher qMatcher2 = qPattern2.matcher(copyDepGraph);
    SemgrexMatcher qMatcher3 = qPattern3.matcher(copyDepGraph);
    SemgrexMatcher qMatcher4 = qPattern4.matcher(copyDepGraph);
    boolean match1 = qMatcher1.find();
    boolean match2 = qMatcher2.find();
    boolean match3 = qMatcher3.find();
    boolean match4 = qMatcher4.find();
    if (match1 || match2 || match3 || match4) {
      List<IndexedWord> wWords = null;
      IndexedWord wWordsRoot = null;
      IndexedWord verb = null;
      IndexedWord subj = null;
      if (match1 || match3) {
        SemgrexMatcher matcher;
        if (match1) {
          matcher = qMatcher1;
        } else {
          matcher = qMatcher3;
        }
        wWordsRoot = matcher.getNode("w");
        wWords = new ArrayList<>(copyDepGraph.getSubgraphVertices(wWordsRoot));
        Collections.sort(wWords);
        verb = matcher.getMatch();
        subj = matcher.getNode("s");
      } else if (match2 || match4) {
        SemgrexMatcher matcher;
        if (match2) {
          matcher = qMatcher2;
        } else {
          matcher = qMatcher4;
        }
        wWords = new ArrayList<>();
        wWordsRoot = matcher.getMatch();
        wWords.add(wWordsRoot);
        verb = matcher.getNode("v");
        subj = matcher.getNode("s");
      }

      List<IndexedWord> subjList = new ArrayList<>(copyDepGraph.getSubgraphVertices(subj));
      Collections.sort(subjList);
      Iterator<IndexedWord> subjIterator = subjList.iterator();
      while (subjIterator.hasNext()) {
        IndexedWord subjWord = subjIterator.next();
        for (SemanticGraphEdge edge : copyDepGraph.getIncomingEdgesSorted(subjWord)) {
          if (edge.getRelation().getShortName().equalsIgnoreCase("advmod")) {
            IndexedWord adverb = edge.getDependent();
            copyDepGraph.removeEdge(edge);
            copyDepGraph.addEdge(verb, adverb, GrammaticalRelation.DEPENDENT, 0, false);
            subjIterator.remove();
            break;
          }
        }
      }
      copyDepGraph.removeVertex(subj);
      copyDepGraph.removeVertex(wWordsRoot);
      SemgrexMatcher qMatcher = qPuncPattern.matcher(copyDepGraph);
      if (qMatcher.find()) {
        copyDepGraph.removeVertex(qMatcher.getMatch());
      }
      ArrayList<String> words = new ArrayList<>();
      for (IndexedWord wWord : wWords) {
        words.add(wWord.word());
      }
      for (IndexedWord subjWord : subjList) {
        words.add(subjWord.word());
      }
      List<IndexedWord> remainingWords = new ArrayList<>(copyDepGraph.getSubgraphVertices(verb));
      Collections.sort(remainingWords);
      for (IndexedWord remainingWord : remainingWords) {
        words.add(remainingWord.word());
      }
      return Optional.of(new Rephrased("I do not know", new Sentence(words), "WQUES"));
    }

    //If the W POS word is the subject it seems safe to not alter the order of the words.
    SemgrexPattern whoSubjPattern = SemgrexPattern.compile("{} >nsubj {pos:/W.*/}");
    SemgrexMatcher whoSubjMatcher = whoSubjPattern.matcher(dependencyGraph);
    if (whoSubjMatcher.find()) {
      return Optional.of(new Rephrased("I do not know", sentence, "WSUBJ"));
    }

    //If it is a "Do ..." or "Does ..." question then output "I do not know if ..."
    if ((originalSentence.lemma(0).equalsIgnoreCase("do") ||
         originalSentence.lemma(0).equalsIgnoreCase("does")) // sometimes does is identified as NNS
             && originalSentence.length() > 1) {
      return Optional.of(new Rephrased("I do not know if", sentence, "DO_QUESTION"));
    }

    //If it is a "Is <word>" or "Are <word>" or "Have <word>" question then flip and output "if <word> is/are ..."
    //If it is a "Can|Could|Would|May|Will <word>" question do the same
    if ((sentence.lemma(0).equalsIgnoreCase("be") ||
         sentence.lemma(0).equalsIgnoreCase("have") ||
         sentence.posTag(0).equals("MD")) && sentence.length() > 1 ) {
      List<String> tokens = sentence.originalTexts();
      List<String> rearranged = new ArrayList<>(tokens);
      rearranged.set(0, tokens.get(1));
      rearranged.set(1, tokens.get(0).toLowerCase());
      return Optional.of(new Rephrased("I do not know if", new Sentence(rearranged), "BE_HAVE_MD_QUESTION"));
    }

    //If it is a "Why ..." question then output "I don't know ..."
    if (sentence.length() > 1 &&
        (sentence.posTag(0).startsWith("W"))) {
      return Optional.of(new Rephrased("I do not know ", sentence, "WQUES_SIMPLE"));
    }

    // Worst Case Scenario
    return Optional.empty();
  }

  protected Sentence replacePronouns(Sentence sentence) {
    //Start at 1 to ignore first I statement in rephrasing
    return replacePronouns(sentence, 0, true);
  }

}
