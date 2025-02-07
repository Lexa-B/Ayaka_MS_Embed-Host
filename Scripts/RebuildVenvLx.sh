chmod u+x "$(dirname "$0")"/**/*.sh
read -p "Delete old virtual environment - Continue (y/n)?" choice
if [ "$choice" = "y" ]; then
  echo "REMOVING...";
  rm -rf ./VenvLx
  read -p "Create new virtual environment - Continue (y/n)?" choice
  if [ "$choice" = "y" ]; then
    echo "BUILDING...";
    python3.12 -m venv ./VenvLx
  fi
fi

if [ -n "$VIRTUAL_ENV" ]; then
  echo "You are in a virtual environment: $VIRTUAL_ENV"
  read -p "Install python requirements - Continue (y/n)?" choice
  if [ "$choice" = "y" ]; then
    echo "INSTALLING...";
    ./Scripts/InstallPyReqs.sh
    echo "."
    echo "."
    echo "."
    echo "."
    echo "."
    echo "."
    echo "."
    echo "================================================"
    echo " ---- Done installing python requirements ----" 
    echo "================================================"
    echo ""
    echo "Installed:"
    pip list
  fi
else
  echo "YOU ARE NOT IN A VIRTUAL ENVIRONMENT"
  echo "Please activate the virtual environment first"
  echo "Please run \"source VenvLx/bin/activate\""
fi




