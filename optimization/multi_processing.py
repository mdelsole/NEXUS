"""
If we have tasks that don't need to be run synchronously, we can split these tasks up.

"""
import multiprocessing

def do_something():
    print("Doing")

# Define the processes


# Not do_something(); we don't want to pass in the return value of the function
p1 = multiprocessing.Process(target=do_something)
p2 = multiprocessing.Process(target=do_something)

# Start the processes:
p1.start()
p2.start()

# Make the process finish before we can move on
# Join = join it back to the main process, so we can use it
p1.join()
p2.join()

print("Finished")
