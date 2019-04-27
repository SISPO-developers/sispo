import bpy

device = "Auto"

cycp = bpy.context.preferences.addons["cycles"].preferences

devices1 = cycp.get_devices()
devices = cycp.devices
device_types = {device.type for device in devices}

if device not in device_types:        
    if "GPU" in device_types:
        device = "GPU"
    else:
        device = "CPU"

print(device_types)
print(device)

# get list of possible values of enum, see http://blender.stackexchange.com/a/2268/599

# pretty print
#lines=[
#("Property", "Value", "Possible Values"),
#("Device Type:", devt, str(devt_list)),
#("Device:", dev, str(dev_list)),
#]
#print("\nGPU compute configuration:")
#for l in lines:
#    print("{0:<20} {1:<20} {2:<50}".format(*l))