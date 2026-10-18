"""Read-only browser smoke check for a running local TeachCopilot instance.

This logs in with the ignored demo credential file, captures teacher and
student views, and optionally sends one harmless student chat question. It
does not save observations, edit profiles, or confirm lesson proposals.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".local" / "ms-playwright"))

from playwright.sync_api import expect, sync_playwright


def login(page, base_url: str, email: str, password: str, timeout_ms: int) -> None:
    page.set_default_timeout(timeout_ms)
    page.goto(base_url, wait_until="networkidle")
    page.get_by_label("Электронная почта").fill(email)
    page.get_by_label("Пароль").fill(password)
    page.get_by_role("button", name="Войти").click()
    expect(page.locator("#workspace")).to_be_visible()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--credentials", type=Path, default=ROOT / ".local" / "rich-demo-credentials.json")
    parser.add_argument("--screenshots", type=Path, default=ROOT / ".local" / "screenshots")
    parser.add_argument("--timeout", type=int, default=90_000, help="browser action timeout in milliseconds")
    parser.add_argument("--skip-chat", action="store_true", help="only check teacher and student dashboards")
    args = parser.parse_args()

    parsed = urlsplit(args.base_url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username or parsed.password:
        parser.error('The UI smoke check accepts only an HTTP loopback URL')

    credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
    teacher = credentials["teacher"]
    students = credentials["students"]
    if not isinstance(teacher, dict) or not students:
        raise SystemExit("credential file must contain teacher and students entries")
    student = students[0]
    args.screenshots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        teacher_page = browser.new_page(viewport={"width": 1440, "height": 1000})
        login(teacher_page, args.base_url, teacher["email"], teacher["password"], args.timeout)
        expect(teacher_page.locator("#role-label")).to_contain_text("Учитель")
        expect(teacher_page.locator("#learner-list .learner-item")).to_have_count(len(students))
        expect(teacher_page.locator("#edit-profile")).to_be_visible()
        teacher_page.screenshot(path=str(args.screenshots / "live-teacher.png"), full_page=True)
        teacher_page.set_viewport_size({"width": 390, "height": 844})
        teacher_page.get_by_role("button", name="Занятие").click()
        teacher_page.screenshot(path=str(args.screenshots / "live-teacher-mobile.png"), full_page=True)

        student_page = browser.new_page(viewport={"width": 1440, "height": 1000})
        login(student_page, args.base_url, student["email"], student["password"], args.timeout)
        expect(student_page.locator("#role-label")).to_contain_text("Ученик")
        expect(student_page.locator("#learner-list .learner-item")).to_have_count(1)
        expect(student_page.locator("#edit-profile")).to_be_hidden()
        expect(student_page.locator("#profile-content")).to_be_visible()
        learner_name = student_page.locator("#learner-name").inner_text()
        student_page.screenshot(path=str(args.screenshots / "live-student.png"), full_page=True)

        chat_status = "skipped"
        if not args.skip_chat:
            mobile_lesson_tab = student_page.get_by_role("button", name="Занятие")
            if mobile_lesson_tab.is_visible():
                mobile_lesson_tab.click()
            student_page.get_by_label("Предмет").select_option("python")
            student_page.get_by_label("Сообщение помощнику").fill("Что покажет print(2 + 3) в Python?")
            with student_page.expect_response(lambda response: response.url.endswith("/v1/chat/completions"), timeout=args.timeout) as response:
                student_page.get_by_role("button", name="Отправить вопрос").click()
            chat_status = str(response.value.status)
            if response.value.status != 200:
                raise RuntimeError('Student chat did not return HTTP 200')
            expect(student_page.locator("#chat-log .chat-bubble.assistant")).to_have_count(2, timeout=args.timeout)
            student_page.screenshot(path=str(args.screenshots / "live-student-chat.png"), full_page=True)
        student_page.set_viewport_size({"width": 390, "height": 844})
        mobile_lesson_tab = student_page.get_by_role("button", name="Занятие")
        if mobile_lesson_tab.is_visible():
            mobile_lesson_tab.click()
        student_page.screenshot(path=str(args.screenshots / "live-student-mobile.png"), full_page=True)

        print(f"Teacher UI: ready; {len(students)} assigned learners; screenshot={args.screenshots / 'live-teacher.png'}")
        print(f"Student UI: ready; own profile={student.get('learner_id', learner_name)}; screenshot={args.screenshots / 'live-student.png'}")
        print(f"Student chat HTTP status: {chat_status}")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
