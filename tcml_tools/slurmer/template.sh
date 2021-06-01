#!/bin/bash

{lines_before}

python3 "{code_dir}/{project}/{file}" \
  {shared_values} \
  {params} \
  "$@"

{lines_after}
