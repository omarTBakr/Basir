# Basir · software engineering

This branch holds Basir's design documents only: user stories, use case diagrams, sequence diagrams and class diagrams, written in PlantUML with each rendered `.png` next to its source. The code lives on the [`phase_0` branch](https://github.com/omarTBakr/Basir/tree/phase_0). This branch is kept separate and is not merged into it.

**Start here:** [`software_engineering/README.md`](software_engineering/README.md) indexes every diagram.

## Rendering

```bash
software_engineering/render.sh                                         # every diagram
software_engineering/render.sh 3_sequence_diagrams/query_where_is.puml # one diagram
```

Needs `java` and graphviz (`dot`). On first run `render.sh` downloads the pinned PlantUML jar into `software_engineering/.tools/` (git-ignored).

## License

**Proprietary — All Rights Reserved.** Copyright (c) 2026 Omar Ayman Bakr. Viewing this repository grants no rights; see [LICENSE](LICENSE). To request permission, contact [@omarTBakr](https://github.com/omarTBakr).
