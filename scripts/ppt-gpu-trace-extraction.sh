#!/bin/bash

display_usage() {
    echo "PPT-GPU Trace Collector: Docker tool to collect traces"
    echo -e "\nRequirements:"
    echo -e "\t- nvidia-docker package needed for the script"
    echo -e "\nUsage:"
    echo -e "\t- ./ppt-gpu-traces [-t|--tool] <PPT-GPU_TOOL_PATH> [-w|--workload] <PATH_TO_THE_EXEC_FOLDER> [-a|--app] <APP_NAME> [-d|--docker] <DOCKER_IMAGE_NAME:DOCKER_TAG> [options]"
    echo -e "\nOptions:"
    echo -e "\t- [-ld  | --leave-docker]              This option is given to allow the user to leave the docker active after executing"
    echo -e "\t- [-h  | --help]              Show this message and exit - it doesn't need any other argument to pass (ex: ./ppt-gpu-traces -h)"
    echo -e "\nExample:"
    echo -e "\t \"./ppt-gpu-traces -t /tools/PPT-GPU -w /workloads/2mm -a 2mm.out -d yarafa/ppt-gpu:traces-cuda11.0\""
    echo -e "\t\t- \"-t /tools/PPT-GPU\" represents the PPT-GPU tool path"
    echo -e "\t\t- \"-w /workloads/2mm\" represents the workloads path where application is located"
    echo -e "\t\t- \"-a 2mm.out\" represents the name of the executable of the application that we will use to collect traces for"
    echo -e "\t\t- \"-d yarafa/ppt-gpu:traces-cuda11.0\" represents the docker image that will be used to run the tool on\n"
}

# detect missing arguments
if [[ (! " $@ " =~ " -h ") && (! " $@ " =~ " --help ") && (((! " $@ " =~ " -t ") && (! " $@ " =~ " --tool ")) || ((! " $@ " =~ " -w ") && (! " $@ " =~ " --workload ")) || ((! " $@ " =~ " -a ") && (! " $@ " =~ " --app ")) || ((! " $@ " =~ " -d ") && (! " $@ " =~ " --docker "))) ]]; then
    echo -e "Error missing argument!\n"
    display_usage
    exit 1
fi


# getting inputs
while [ -n "$1" ]; do

    case "$1" in

    # check whether user had supplied -h or --help . If yes display usage
    -h)
        display_usage
        exit 0
        shift
        ;;

    --help)
        display_usage
        exit 0
        shift
        ;;

    ## -- Process Options --

    --tool)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no tool path was defined!\n"
            display_usage
            exit 1
        fi
        TOOL=$2
        shift
        ;;

    -t)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no tool path was defined!\n"
            display_usage
            exit 1
        fi
        TOOL=$2
        shift
        ;;

    --workload)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no workload path was defined!\n"
            display_usage
            exit 1
        fi
        WORKLOAD=$2
        shift
        ;;

    -w)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no workload path was defined!\n"
            display_usage
            exit 1
        fi
        WORKLOAD=$2
        shift
        ;;

    --app)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no application executable was defined!\n"
            display_usage
            exit 1
        fi
        APP=$2
        shift
        ;;

    -a)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no application executable was defined!\n"
            display_usage
            exit 1
        fi
        APP=$2
        shift
        ;;

    --docker)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no docker image was defined!\n"
            display_usage
            exit 1
        fi
        DOCKER=$2
        shift
        ;;

    -d)
        if [[ (-z "$2") || "$2" == "-"* ]]; then
            echo -e "Error no docker image was defined!\n"
            display_usage
            exit 1
        fi
        DOCKER=$2
        shift
        ;;

    --leave-docker)
        LEAVE_DOCKER=1
        shift
        ;;

    -ld)
        LEAVE_DOCKER=1
        shift
        ;;

    -*)
        echo "Option \"$1\" not recognized"
        display_usage
        exit 1
        ;;

    *)
        shift
        ;;

    esac
    shift

done

DOCKER_PATH=/docker
DOCKER_WORKLOAD_PATH="${DOCKER_PATH}/workload"
DOCKER_TOOL_PATH="${DOCKER_PATH}/tool"
DOCKER_TOOL_EXEC_PATH=$DOCKER_PATH/tool/tracing_tool/tracer.so

UID=$(id -u)
GID=$(id -g)

echo "Loading the docker container......"
docker run --rm --name traces-ppt-gpu -tid --user $UID:$GID --env LD_PRELOAD=$DOCKER_TOOL_EXEC_PATH --gpus device=0 -v $WORKLOAD:$DOCKER_WORKLOAD_PATH -v $TOOL:$DOCKER_TOOL_PATH -w $DOCKER_WORKLOAD_PATH --entrypoint /bin/bash $DOCKER  > /dev/null

echo "Executing the tool inside the docker....."
docker exec traces-ppt-gpu ./$APP

if [[ -z $LEAVE_DOCKER ]]; then
    echo "Deleting the docker container..."
    docker container rm -f traces-ppt-gpu > /dev/null
fi

echo -e "Tracing Completed!\n"
