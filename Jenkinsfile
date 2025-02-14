pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'my-dl-model:latest'
        DOCKER_HUB_REPO = 'mydockerhubuser/my-dl-model'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Onyesi-john/dlops_testing.git'
            }
        }

        stage('Verify Python Installation') {
             steps {
                  sh 'python3 --version'  // Check if Python 3 is installed
            }
        }


        stage('Set Up Python Environment') {
            steps {
               script {
            // Create and activate the virtual environment
            sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
            '''
                }
             }
          }


        stage('Train Model') {
            steps {
                sh 'python model/train.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKER_IMAGE .'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withDockerRegistry([credentialsId: 'docker-hub-credentials', url: '']) {
                    sh 'docker tag $DOCKER_IMAGE $DOCKER_HUB_REPO:latest'
                    sh 'docker push $DOCKER_HUB_REPO:latest'
                }
            }
        }

        

        stage('Monitor Model Performance') {
            steps {
                sh 'curl http://localhost:9090/api/v1/query?query=loss'
            }
        }

        stage('Trigger Auto-Retraining') {
            steps {
                script {
                    def loss = sh(script: "python model/check_loss.py", returnStdout: true).trim()
                    if (loss.toDouble() > 0.5) {
                        sh 'python model/retrain.py'
                    }
                }
            }
        }
    }
}
