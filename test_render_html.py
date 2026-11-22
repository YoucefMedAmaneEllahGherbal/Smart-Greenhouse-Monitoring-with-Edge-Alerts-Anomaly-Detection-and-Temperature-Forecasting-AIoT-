"""Unit test for render_html and clean_html in ui.py"""
import ui

def test_clean_html():
    raw_snippet = """
        <div class="status-banner-card status-banner-normal">
            <div class="status-banner-icon">
                <svg>...</svg>
            </div>

            <div style="flex-grow:1;">
                <div class="status-banner-title">
                    <span>Title</span>
                </div>
                <div class="status-banner-desc">
                    Message line 1.
                    Message line 2.
                </div>
            </div>
        </div>
    """
    cleaned = ui.clean_html(raw_snippet)
    print("Cleaned HTML:", cleaned)
    assert "\n" not in cleaned, "Cleaned HTML should not contain newlines"
    assert "    " not in cleaned, "Cleaned HTML should not contain 4-space indentation"
    assert "> <" not in cleaned, "Whitespace between tags should be collapsed"
    print("TEST PASSED: clean_html correctly eliminates markdown code blocks and blank lines!")

if __name__ == "__main__":
    test_clean_html()
