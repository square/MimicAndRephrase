package ai.eloquent.rephrasing;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.semgrex.SemgrexMatcher;
import edu.stanford.nlp.semgraph.semgrex.SemgrexPattern;
import edu.stanford.nlp.simple.Sentence;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.Pair;

import java.util.*;

public class RuleBasedIDKRephraser extends RuleBasedRephraser {

  public Optional<String> rephrased(Sentence request) {
    //Preprocess the sentence by switching the point of view of pronouns i.e. I to you
    Optional<Pair<String,Sentence>> rephrasedPair = rephraseQuestion(request);
    if (rephrasedPair.isPresent()) {
      //Get initial phrase
      String start = rephrasedPair.get().first;

      //Make sure not upper cased
      Sentence rephrased = rephrasedPair
              .map(x -> removeQuestionMark(x.second))
              .map(x -> adjustCapitalization(x))
              .map(x -> replacePronouns(x)).get();

      //Let's ignore the condensed flag
      return Optional.of(start + " " + rephrased);
    } else {
      return Optional.empty();
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


  private Optional<Pair<String,Sentence>> rephraseQuestion(Sentence sentence) {
    //Get graph for semgrex searches
    SemanticGraph dependencyGraph = sentence.dependencyGraph();

    //Check for imperative sentence (no subject, no W pos, no question mark)
    SemgrexPattern subjPattern = SemgrexPattern.compile("{} >nsubj {}");
    SemgrexPattern wPattern = SemgrexPattern.compile("{pos:/W.*/}");
    SemgrexPattern qPuncPattern = SemgrexPattern.compile("{word:/\\?/}");
    // Handles sentences like "Help me"
    if (!subjPattern.matcher(dependencyGraph).find() && !wPattern.matcher(dependencyGraph).find() && !qPuncPattern.matcher(dependencyGraph).find()) {
      return Optional.of(Pair.makePair("I do not know how to", sentence));
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
      return Optional.of(Pair.makePair("I do not know", new Sentence(words)));
    }

    //If the W POS word is the subject it seems safe to not alter the order of the words.
    SemgrexPattern whoSubjPattern = SemgrexPattern.compile("{} >nsubj {pos:/W.*/}");
    SemgrexMatcher whoSubjMatcher = whoSubjPattern.matcher(dependencyGraph);
    if (whoSubjMatcher.find()) {
      return Optional.of(Pair.makePair("I do not know", sentence));
    }

    //If it is a "Do ..." or "Does ..." question then output "I do not know if ..."
    if (sentence.lemma(0).toLowerCase().equals("do") && sentence.length() > 1) {
      return Optional.of(Pair.makePair("I do not know if", sentence));
    }

    //If it is a "Is <word>" or "Are <word>" question then flip "if <word> is/are ..."
    //If it is a "Can <word>" or "May <word>" question do the same
    if ((sentence.lemma(0).toLowerCase().equals("be") ||
        sentence.lemma(0).toLowerCase().equals("can") ||
        sentence.lemma(0).toLowerCase().equals("may")) && sentence.length() > 1) {
      List<String> tokens = sentence.originalTexts();
      List<String> rearranged = new ArrayList<>(tokens);
      String firstToken = tokens.get(0);
      rearranged.set(0, tokens.get(1));
      rearranged.set(1, firstToken);
      return Optional.of(Pair.makePair("I do not know if", new Sentence(rearranged)));
    }

    //Worst Case Scenario
    return Optional.empty();
  }

  protected Sentence replacePronouns(Sentence sentence) {
    //Start at 1 to ignore first I statement in rephrasing
    return replacePronouns(sentence, 0, true);
  }

}
