## graphify

This project has a knowledge graph at graphify-out/ with god nodes,
community structure, and cross-file relationships.

### Rules

- For ANY codebase question or task, if graphify-out/graph.json exists,
  use Graphify FIRST before broad source-file exploration.
- Start with:
  `graphify query "<question>"`
- Use `graphify explain "<concept>"` when investigating a specific
  class, function, module, component, service, or concept.
- Use `graphify path "<A>" "<B>"` when determining how two concepts,
  files, modules, classes, or functions are connected.
- Do NOT use broad grep, recursive file searching, or mass source-file
  reading before using Graphify when the graph is available.
- After Graphify identifies the relevant scope, read only the source
  files required to verify implementation details.
- Treat EXTRACTED relationships as directly supported by source code
  and INFERRED relationships as hypotheses that should be verified
  against source when correctness matters.
- If graphify-out/wiki/index.md exists, use it for broad project
  navigation instead of browsing the entire source tree.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review
  or when query/path/explain do not provide enough context.
- After modifying code, run:
  `graphify update .`
  to keep the graph current.
- Never refuse to read source code merely because Graphify was used.
  Graphify is for efficient discovery and context reduction; source
  code remains the final authority for implementation details.