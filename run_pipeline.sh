#!/usr/bin/env bash
# ==============================================================================
# Reproducible Pipeline Runner: Distributed Bias Mitigation on Apache Spark
# ==============================================================================
# This script sets up the environment, downloads datasets if not present,
# runs correctness benchmarks, runs scalability benchmarks, and generates plots.
#
# Usage:
#   ./run_pipeline.sh          # Full execution (up to 1,000,000 rows)
#   ./run_pipeline.sh --quick  # Fast demo execution (up to 100,000 rows)
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================================================"
echo " Starting Reproducible Pipeline: Distributed Bias Mitigation on Spark"
echo "========================================================================"

# 1. Java Environment Discovery
if [ -z "$JAVA_HOME" ]; then
    if [ -d "/usr/lib/jvm/java-21-openjdk-amd64" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-21-openjdk-amd64"
    elif [ -d "/usr/lib/jvm/java-17-openjdk-amd64" ]; then
        export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
    elif [ -d "/usr/lib/jvm/default-java" ]; then
        export JAVA_HOME="/usr/lib/jvm/default-java"
    fi
fi

if [ -n "$JAVA_HOME" ]; then
    echo "[OK] Found JAVA_HOME=$JAVA_HOME"
else
    echo "[WARNING] JAVA_HOME not detected. Spark will use default system java."
fi

# 2. Virtual Environment Setup
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt
else
    echo "[OK] Python virtual environment already exists at $VENV_DIR"
fi

PYTHON_BIN="$VENV_DIR/bin/python"

# 3. Automatic Dataset Download (if not present)
DATA_DIR="$PROJECT_DIR/data"
ADULT_FILE="$DATA_DIR/adult.csv"
COMPAS_FILE="$DATA_DIR/compas.csv"
FOLK_FILE="$DATA_DIR/folktables_acs.csv"

if [ ! -f "$ADULT_FILE" ] || [ ! -f "$COMPAS_FILE" ] || [ ! -f "$FOLK_FILE" ]; then
    echo "[2/5] Datasets missing in data/ - Downloading UCI Adult, COMPAS, and Folktables..."
    "$PYTHON_BIN" "$DATA_DIR/download_datasets.py"
else
    echo "[OK] All required datasets detected in data/ (no re-download needed)"
fi

# 4. Correctness Benchmarks (UCI Adult & COMPAS)
echo "[3/5] Running Correctness Benchmarks on UCI Adult & COMPAS..."
"$PYTHON_BIN" "$PROJECT_DIR/run_correctness.py"

# 5. Scalability Benchmarks (Folktables ACS PUMS)
echo "[4/5] Running Scalability Benchmarks on Folktables..."
if [ "$1" == "--quick" ]; then
    echo "  (Quick mode enabled: testing scales 50K and 100K rows)"
    "$PYTHON_BIN" -c "
from bias_mitigation.spark_utils import get_spark_session
from run_scalability import run_scalability_experiment
spark = get_spark_session('QuickScalability')
run_scalability_experiment(spark, scales=[50000, 100000])
spark.stop()
"
else
    "$PYTHON_BIN" "$PROJECT_DIR/run_scalability.py"
fi

# 6. Generate Figures
echo "[5/5] Generating Visualization Figures..."
"$PYTHON_BIN" "$PROJECT_DIR/plot_results.py"

echo "========================================================================"
echo " Pipeline Finished Successfully!"
echo " Results and plots generated in: $PROJECT_DIR/output/"
echo "   - $PROJECT_DIR/output/scalability_curves.png"
echo "   - $PROJECT_DIR/output/fairness_accuracy_tradeoff.png"
echo "   - $PROJECT_DIR/output/correctness_results.json"
echo "   - $PROJECT_DIR/output/scalability_results.json"
echo "========================================================================"
