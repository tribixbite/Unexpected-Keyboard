package juloo.keyboard2;

import java.util.List;

/** Keep track of the word being typed. */
public final class CurrentlyTypedWord
{
  public CurrentlyTypedWord()
  {
  }

  public static interface Callback
  {
    public void currently_typed_word(String word);
  }
}
