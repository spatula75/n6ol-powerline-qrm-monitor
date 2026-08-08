# Agentic AI Usage in this Project

## Introduction

Agentic "AI" use - in this case, using large language models and tools to develop and test software, remains controversial.  This project has made considerable use of AI, but in a careful and principled manner.  Care has been taken to avoid the generation of "slop" as is tragically common in so many fly-by-night projects.

## Principles

### Implementation, not design

"AI" tools I've found are reasonably good at producing units of code under supervision.  They also do a decent job of refactoring existing code to more closely adhere to established patterns, or under direction to patterns desired by the developer.  In a sense, they can be used very effectively as "fancy auto-complete" and "fancy refactor" tools, a logical extension from tools available in every IDE for decades.

Where I find them lacking is in the imagination required to establish and uphold good design principles, and this, I believe, is why so many projects that are effectively *led* by "AI" tools so quickly turn into a giant pile of unmaintainable slop.

Good software development practices require sitting down and thinking about how code *should* be structured, and occasionally inspecting where things are going within a codebase with an eye for re-structuring or re-designing when the time is right.  "AI" agents, I find, never or almost never make suggestions like, "hey, this looks like it might be better served by a pipes and filters pattern..."

Thus in this project (and all my projects), "AI" is for implementation, never design.  It is a servant carrying out my wishes so that I don't have to type so much.

### Don't Overdrive Your Headlights

When I was learning to drive a car, one of the admonishments we were given in Driver's Ed in a discussion about driving at night was "don't overdrive your headlights," meaning, don't drive at a speed such that you won't have time to react to an object from the time it first reaches the outer reach of your headlights and the time your car reaches it.

A similar principle can be applied to the use of agentic "AI" in two ways.  One, don't churn out code so fast that you don't have time to understand it and review it to make sure that the "AI" is not making a hot mess of your codebase.  Two, and this is important, *don't ask "AI" to code anything that you couldn't code yourself*.

That's worth saying again: *don't ask "AI" to code anything that you couldn't code yourself*.

You need to be *able* to review the code, which means you need to be able to *understand* the code, and you can only *understand* the code if you were capable of writing it yourself.

Fortunately, I've been developing software for over three decades, and I've dabbled sufficiently in DSP over the years that I was able to write the first pass at this application myself, before "AI" tools even existed.  And while I'm definitely not *good* at UI development, it's something I *can* do if it becomes absolutely necessary.

### End-User Documentation: never AI

For technical documentation, comments, docstrings, etc., LLMs can be persuaded to produce prose I might characterize as "basically acceptable" even if it isn't great.  When coerced to follow the Simplified Technical English rules, the language is passable.

For end-user-facing documentation?  Absolutely not.  Every word of these docs has been written by hand, by a human (me) for other humans (you).  No giant walls of textslop here.  LLMs are simply too inept at producing prose that flows naturally and communicates ideas clearly.

### Conclusion

Agentic "AI" has been and will continue to be a tool used by this project to aid in controlled, thoughtful development, under the guidelines above.  I understand there will be some for whom this is reflexively intolerable.  They are invited to seek out other projects or start their own.
