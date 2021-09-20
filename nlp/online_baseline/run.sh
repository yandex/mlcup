set -e 
echo "Running!"
# set this env variable so we can use the modules installed in setup.sh
export PYTHONUSERBASE=$(realpath modules)

python3 online_baseline.py $INPUT_PATH $OUTPUT_PATH --data $SOLUTION_ROOT/data.pkl --embeddings $DATA_ROOT/embeddings_with_lemmas.npz --tokenizer $DATA_ROOT/trained_roberta/

echo "Finished"
