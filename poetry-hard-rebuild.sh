# Stop the current virtualenv if active or alternative use
# `exit` to exit from a Poetry shell session
deactivate

# Remove all the files of the current environment of the folder we are in
POETRY_LOCATION=`poetry env info -p`
echo "Poetry is $POETRY_LOCATION"
rm -rf "$POETRY_LOCATION"

# Reactivate Poetry shell
poetry shell

# Install everything
poetry install
