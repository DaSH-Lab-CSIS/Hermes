# These five lines are all you need for the worker script.
import sys
sys.path.append('../hermes/')
import HermesWorker

worker = HermesWorker.HermesWorker()
worker.beginTraining()
