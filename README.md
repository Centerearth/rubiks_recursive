Attempting to work through coding a rubik's cube recursively by automating most of the writing-code part using Gemini pro. First attempt to "vibe code." I set myself the simple rule: anything I propose, Gemini implements, anything Gemini proposes, I implement. Final solution works but is incredibly slow and only works in good time for very shallow solves, since solving a cube that for example would take 20 moves would require 12^20 moves.

Edit:
Since implementing a bi-dreictional search, the solver can now handle solves up to 14 moves within a couple of minutes.
