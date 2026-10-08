import Foundation

// Drive the production slot assembler without opening any audio device.
let input = try JSONSerialization.jsonObject(with: FileHandle.standardInput.readDataToEndOfFile()) as! [String: Any]
let channels = input["channels"] as! Int
let framesPerSlot = input["frames_per_slot"] as! Int
let batches = input["batches"] as! [[String: [NSNumber]]]
var system: [Float] = []
var microphone: [Float] = []
var results: [[[Float]]] = []
for batch in batches {
    system.append(contentsOf: batch["sys"]!.map { $0.floatValue })
    microphone.append(contentsOf: batch["mic"]!.map { $0.floatValue })
    var slots: [[Float]] = []
    drainCaptureBuffers(system: &system, microphone: &microphone, channels: channels,
                        framesPerSlot: framesPerSlot, slotCount: 16) { slots.append(Array($0)) }
    results.append(slots)
}
FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: results))
