import AppKit

let repoRoot = Bundle.main.bundleURL.resolvingSymlinksInPath()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

func readEnvValue(_ key: String, fallback: String) -> String {
    let envFile = repoRoot.appendingPathComponent(".env")
    guard let content = try? String(contentsOf: envFile, encoding: .utf8) else { return fallback }
    for line in content.components(separatedBy: .newlines) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.hasPrefix("#") else { continue }
        let parts = trimmed.components(separatedBy: "=")
        guard parts.count >= 2 else { continue }
        guard parts[0].trimmingCharacters(in: .whitespaces) == key else { continue }

        var val = parts.dropFirst().joined(separator: "=").trimmingCharacters(in: .whitespaces)
        if let commentIndex = val.firstIndex(of: "#") {
            val = String(val[..<commentIndex]).trimmingCharacters(in: .whitespaces)
        }
        val = val.replacingOccurrences(of: "\"", with: "").replacingOccurrences(of: "'", with: "")
        return val.isEmpty ? fallback : val
    }
    return fallback
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var backend: Process?
    var port = "8000"
    var isQuitting = false
    var captureEngine: CaptureEngine?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Single instance guard
        let bundleID = Bundle.main.bundleIdentifier ?? "com.resonance.app"
        if NSRunningApplication.runningApplications(withBundleIdentifier: bundleID).count > 1 {
            NSApp.terminate(nil)
            return
        }

        port = readEnvValue("RESONANCE_PORT", fallback: "8000")

        setupStatusItem()
        startBackend()

        // Initialize SCK and IPC
        captureEngine = CaptureEngine()
    }

    private func createMenuItem(title: String, action: Selector, key: String, symbolName: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
        if let symbolImage = NSImage(systemSymbolName: symbolName, accessibilityDescription: title) {
            item.image = symbolImage
        }
        return item
    }

    func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            guard let image = NSImage(named: "StatusBarIcon") else {
                fatalError("[Resonance] Missing required bundle resource: StatusBarIcon")
            }
            image.isTemplate = true
            button.image = image
        }

        let menu = NSMenu()
        menu.addItem(createMenuItem(title: "Open in Browser", action: #selector(openBrowser), key: "o", symbolName: "globe"))
        menu.addItem(NSMenuItem.separator())

        menu.addItem(createMenuItem(title: "Settings", action: #selector(openSettings), key: ",", symbolName: "gearshape"))
        menu.addItem(createMenuItem(title: "Show Logs", action: #selector(showLogs), key: "l", symbolName: "doc.text"))
        menu.addItem(createMenuItem(title: "Restart Backend", action: #selector(restartBackend), key: "r", symbolName: "arrow.clockwise"))

        menu.addItem(NSMenuItem.separator())
        menu.addItem(createMenuItem(title: "Quit", action: #selector(quit), key: "q", symbolName: "power"))

        statusItem.menu = menu
    }

    func startBackend() {
        // Uses /usr/bin/env so `just` is resolved via PATH (works for both Intel and Apple Silicon)
        backend = Process()
        backend?.executableURL = URL(fileURLWithPath: "/bin/zsh")
        backend?.arguments = ["-l", "-c", "export PATH=\"/opt/homebrew/bin:$HOME/.cargo/bin:$PATH\"; exec just dev"]
        backend?.currentDirectoryURL = repoRoot

        let errorPipe = Pipe()
        backend?.standardError = errorPipe

        backend?.terminationHandler = { [weak self] process in
            guard let self = self, !self.isQuitting else { return }
            if process.terminationStatus != 0 {
                let errorData = (try? errorPipe.fileHandleForReading.readToEnd()) ?? Data()
                let errorMessage = String(data: errorData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "No stderr output"

                DispatchQueue.main.async {
                    let alert = NSAlert()
                    alert.messageText = "Backend Process Failed (Code \(process.terminationStatus))"
                    alert.informativeText = "Details:\n\(errorMessage.isEmpty ? "Unknown error (check logs)" : errorMessage)"
                    alert.alertStyle = .critical
                    alert.addButton(withTitle: "Quit Resonance")

                    NSApp.activate(ignoringOtherApps: true)
                    alert.runModal()
                    NSApp.terminate(nil)
                }
            }
        }

        do {
            try backend?.run()
        } catch {
            NSLog("[Resonance] Failed to start backend: \(error)")
        }
    }

    @objc func openBrowser() {
        guard let url = URL(string: "http://localhost:\(port)") else { return }
        NSWorkspace.shared.open(url)
    }

    @objc func openSettings() {
        let envPath = repoRoot.appendingPathComponent(".env").path
        let task = Process()
        task.launchPath = "/usr/bin/open"
        task.arguments = ["-e", envPath] // -e forces opening in TextEdit
        try? task.run()
    }

    @objc func showLogs() {
        let customLogFile = readEnvValue("RESONANCE_LOG_FILE", fallback: "")
        if !customLogFile.isEmpty {
            let expandedPath = (customLogFile as NSString).expandingTildeInPath
            let logURL = URL(fileURLWithPath: expandedPath)
            let logDir = logURL.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
            NSWorkspace.shared.open(logDir)
        } else {
            let logDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Logs/Resonance")
            try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
            NSWorkspace.shared.open(logDir)
        }
    }

    private func terminateProcessGroup(_ process: Process) {
        let pid = process.processIdentifier
        guard pid > 0 else { return }

        // Invariants: Foundation.Process places the child in an isolated process group (PGID == PID).
        // Signaling the negative PGID ensures intermediate shells and spawned Python workers
        // are torn down reliably without abandoning orphaned processes that hold network ports.
        kill(-pid, SIGTERM)
        kill(pid, SIGTERM)

        for _ in 0..<15 {
            if !process.isRunning { return }
            Thread.sleep(forTimeInterval: 0.1)
        }

        if process.isRunning {
            kill(-pid, SIGKILL)
            kill(pid, SIGKILL)
        }
    }

    @objc func restartBackend() {
        DispatchQueue.global().async {
            // Workaround: Mutate state before termination so terminationHandler ignores the exit code
            self.isQuitting = true

            if let process = self.backend, process.isRunning {
                self.terminateProcessGroup(process)
            }

            DispatchQueue.main.async {
                self.isQuitting = false
                self.port = readEnvValue("RESONANCE_PORT", fallback: "8000")
                self.startBackend()
            }
        }
    }

    @objc func quit() {
        NSApp.terminate(nil)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let process = backend, process.isRunning else {
            return .terminateNow
        }

        isQuitting = true

        DispatchQueue.global().async {
            self.terminateProcessGroup(process)

            DispatchQueue.main.async {
                NSApp.reply(toApplicationShouldTerminate: true)
            }
        }

        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {}
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
