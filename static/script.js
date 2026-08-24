(function () {
    "use strict";

    var form = document.getElementById("predictForm");
    var issue = document.getElementById("issue");
    var submitBtn = document.getElementById("submitBtn");
    var themeToggle = document.getElementById("themeToggle");

    function getTheme() {
        return localStorage.getItem("helpdesk-theme") || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    }

    function setTheme(theme) {
        document.body.classList.remove("light", "dark");
        document.body.classList.add(theme);
        document.documentElement.setAttribute("data-theme", theme === "dark" ? "dark" : "");
        localStorage.setItem("helpdesk-theme", theme);
    }

    function initTheme() {
        setTheme(getTheme());
    }

    if (themeToggle) {
        themeToggle.addEventListener("click", function () {
            var next = getTheme() === "dark" ? "light" : "dark";
            setTheme(next);
        });
    }
    initTheme();

    var samples = document.querySelectorAll(".btn-sample");
    samples.forEach(function (btn) {
        btn.addEventListener("click", function () {
            var text = this.getAttribute("data-text");
            if (issue && text) {
                issue.value = text;
                issue.focus();
            }
        });
    });

    function setLoading(loading) {
        if (!submitBtn) return;
        if (loading) {
            submitBtn.classList.add("loading");
            submitBtn.disabled = true;
        } else {
            submitBtn.classList.remove("loading");
            submitBtn.disabled = false;
        }
    }

    if (form) {
        form.addEventListener("submit", function (e) {
            var value = (issue && issue.value || "").trim();
            if (!value) {
                e.preventDefault();
                if (issue) {
                    issue.focus();
                    issue.placeholder = "Please describe your issue…";
                }
                return;
            }
            setLoading(true);
        });
    }

    if (issue) {
        issue.addEventListener("input", function () {
            if (this.placeholder === "Please describe your issue…") {
                this.placeholder = "e.g. Cannot connect to WiFi, Printer not responding...";
            }
        });
    }
})();
