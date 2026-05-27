USER = ioana_calina.pascu
REMOTE = $(USER)@fep.grid.pub.ro
REMOTE_DIR = /export/home/acs/stud/i/$(USER)/SSL_NLP_project

TAR_EXCLUDE = --exclude='.git' --exclude='*.sif' --exclude='__pycache__' \
              --exclude='*.pyc' --exclude='data' --exclude='.DS_Store' \
              --exclude='.venv' --exclude='*.jsonl' --exclude='results' \
              --exclude='slurm_outs' --exclude='CLAUDE.md' --exclude='old_results'

SSH_QUIET = ssh -q -o LogLevel=QUIET

ACCOUNT = student

# =============================================================================
# SSL QUICK REFERENCE
# =============================================================================
.PHONY: ssl-help ssl-setup
ssl-help:
	@echo "=========================================="
	@echo "SSL Implementation - Quick Commands"
	@echo "=========================================="
	@echo "make ssl-setup       - Create sample corpus"
	@echo "make prepare-ssl-corpus - Create corpus from synthetic.csv"
	@echo "make train-ssl       - Train DAE on cluster"
	@echo ""
	@echo "See QUICK_REFERENCE_SSL.md for more!"
	@echo "=========================================="

ssl-setup: prepare-ssl-corpus
	@echo "✓ Sample corpus ready"
	@echo "Try: python3 src/ssl_trainer.py --unlabeled_data data/unlabeled_corpus_sample.txt --out_dir results/ssl_dae_test --epochs 1 --batch_size 4"

# =============================================================================
# LOCAL COMMANDS
# =============================================================================
upload:
	tar $(TAR_EXCLUDE) -czf /tmp/ssl_nlp_upload.tar.gz .
	$(SSH_QUIET) $(REMOTE) "mkdir -p $(REMOTE_DIR)"
	scp -q /tmp/ssl_nlp_upload.tar.gz $(REMOTE):$(REMOTE_DIR)/upload.tar.gz
	$(SSH_QUIET) $(REMOTE) "cd $(REMOTE_DIR) && tar -xzf upload.tar.gz && rm upload.tar.gz"
	rm /tmp/ssl_nlp_upload.tar.gz
	@echo "upload done."

upload-data:
	$(SSH_QUIET) $(REMOTE) "mkdir -p $(REMOTE_DIR)/data"
	scp -q ./data/synthetic.csv $(REMOTE):$(REMOTE_DIR)/data/synthetic.csv
	@echo "data uploaded."

download:
	$(SSH_QUIET) $(REMOTE) "cd $(REMOTE_DIR) && tar --exclude='*.sif' --exclude='__pycache__' --exclude='data' --exclude='results/**/*.pt' -czf /tmp/ssl_nlp_dl.tar.gz ."
	scp -q $(REMOTE):/tmp/ssl_nlp_dl.tar.gz /tmp/ssl_nlp_dl.tar.gz
	$(SSH_QUIET) $(REMOTE) "rm /tmp/ssl_nlp_dl.tar.gz"
	tar -xzf /tmp/ssl_nlp_dl.tar.gz
	rm /tmp/ssl_nlp_dl.tar.gz
	@echo "download done."

results:
	mkdir -p results
	$(SSH_QUIET) $(REMOTE) "cd $(REMOTE_DIR) && tar --exclude='*.pt' --exclude='*.bin' --exclude='*.model' -czf /tmp/ssl_nlp_results.tar.gz results"
	scp -q $(REMOTE):/tmp/ssl_nlp_results.tar.gz /tmp/ssl_nlp_results.tar.gz
	$(SSH_QUIET) $(REMOTE) "rm /tmp/ssl_nlp_results.tar.gz"
	tar -xzf /tmp/ssl_nlp_results.tar.gz
	rm /tmp/ssl_nlp_results.tar.gz
	@echo "results downloaded."

# =============================================================================
# FEP COMMANDS
# =============================================================================
prep:
	sbatch -A $(ACCOUNT) scripts/prep.sh

# full corpus extraction from data/synthetic.csv via sbatch
prep-ssl-corpus:
	sbatch -A $(ACCOUNT) scripts/prep_ssl_corpus.sh

train-ssl:
	sbatch -A $(ACCOUNT) scripts/train_ssl.sh

train-detector:
	sbatch -A $(ACCOUNT) scripts/train_detector.sh

train-seq2seq:
	sbatch -A $(ACCOUNT) scripts/train_seq2seq.sh

eval-syn:
	sbatch -A $(ACCOUNT) scripts/eval_syn.sh

eval:
	sbatch -A $(ACCOUNT) scripts/eval.sh

eval-rescore:
	sbatch -A $(ACCOUNT) scripts/eval_rescore.sh

sweep-threshold:
	sbatch -A $(ACCOUNT) scripts/sweep_threshold.sh

sweep-lambda:
	sbatch -A $(ACCOUNT) scripts/sweep_lambda.sh

demo:
	sbatch -A $(ACCOUNT) scripts/demo.sh

download-models:
	mkdir -p $$HOME/.cache/huggingface
	singularity exec --env HF_HOME=$$HOME/.cache/huggingface --env HF_TOKEN=$$(cat $$HOME/.hf_token 2>/dev/null) $$HOME/ml_general_v5.sif python3 download_models.py

status:
	squeue -u $$USER

cancel:
	scancel $$(squeue -h -u $$USER -o "%i" | head -n1)

clean-results:
	rm -rf results/

clean-slurm:
	rm -f slurm-*.out job_*.txt