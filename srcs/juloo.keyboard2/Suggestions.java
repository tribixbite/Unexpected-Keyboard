package juloo.keyboard2;

import java.util.List;

/** Keep track of the word being typed and provide suggestions for
    [CandidatesView]. */
public final class Suggestions
{
  Callback _callback;

  public Suggestions(Callback c)
  {
    _callback = c;
  }

  public currently_typed_word(String word)
  {
    // TODO
  }

  public static interface Callback
  {
    public void set_suggestions(List<String> suggestions);
  }
}
