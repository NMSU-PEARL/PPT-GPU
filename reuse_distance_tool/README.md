
The ***reuse_distance_tool*** is based on the available code of the following paper: Q. Niu et al., "PARDA: A Fast Parallel Reuse Distance Analysis Algorithm", (IPDPS'12)


## Building the code 

* USING DOCKER: you can skip steps below and diectly build the tool through the docker image uaing the following command:
        ```
        docker run --rm -v $(pwd):$(pwd) -w $(pwd) yarafa/ppt-gpu:simulation-latest make clean && make
        ```

* Step 1: parda use ***glib*** standard linux library.    
        - Ubuntu system, execute following sudo command:
                ```
                sudo apt-get install libglib2.0-dev
                ```  
        - Centos system, execute following sudo command: 
        ```
        sudo yum install glib*
        ```
        
* Step 2: make clean && make

It will output parda.x executable file that is called automatically within the main framework 

