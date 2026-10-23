(() => {
  'use strict';
  const API = '';
  const tokenKey = 'teachcopilot.session';
  const greeting = 'Привет! Расскажи, что разбираете, и я помогу найти следующий шаг.';
  const state = { token: sessionStorage.getItem(tokenKey), account: null, learners: [], learnerId: null, dashboard: null, topics: [], task: null, proposal: null, revision: null, chatByLearner: new Map(), nextStepsByLearner: new Map(), learningSessionsByLearner: new Map(), pendingMessages: new Map(), chatTopicsByLearnerSubject: new Map(), memoryCorrectionJobs: new Map(), memoryViewRevision: 0, viewRevision: 0 };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const el = (tag, cls, value) => { const node = document.createElement(tag); if (cls) node.className = cls; if (value !== undefined) node.textContent = value; return node; };
  const cleanFictionalPrefix = text => String(text || '').trim().replace(/^(?:DEMO|Демо|вымышленный)(?:\s*[–\u2014-]\s*|\s+)/iu, '').trim();
  const initials = text => cleanFictionalPrefix(text).split(/[\s–\u2014-]+/).filter(Boolean).slice(0, 2).map(part => part[0] || '').join('').toLocaleUpperCase('ru') || '?';
  const languageLabel = value => ({ russian: 'Русский', ru: 'Русский', english: 'Английский', en: 'Английский', spanish: 'Испанский', es: 'Испанский' })[String(value || '').toLocaleLowerCase('ru')] || String(value || '');
  const displayName = account => account?.display_name || account?.email || 'Пользователь';
  const errorText = error => error?.message || 'Попробуйте ещё раз.';
  let toastTimer;
  let loginBusy = false;
  const guidedContexts = new Map();
  const sessionRecovery = new Map();
  const sessionClosures = new Map();
  let learningArchive = null;
  let guidedViewRevision = 0;
  let chatReturnAttempt = null;
  const guidedTopic = 'math.g4.numbers.division_by_1_2_digit';
  const guidedTopicTitle = 'Деление на однозначное и двузначное число';

  async function request(path, options = {}) {
    const requestToken = state.token;
    const headers = new Headers(options.headers || {});
    if (requestToken) headers.set('Authorization', `Bearer ${requestToken}`);
    if (options.body !== undefined) headers.set('Content-Type', 'application/json');
    let response;
    try { response = await fetch(`${API}${path}`, { ...options, headers, body: options.body === undefined ? undefined : JSON.stringify(options.body) }); }
    catch { throw new Error('Нет связи с локальным сервером. Проверьте, что приложение запущено.'); }
    if (response.status === 401 && requestToken && requestToken === state.token) { signOutLocal(); showLogin('Сеанс завершён. Войдите снова.'); throw new Error('Сеанс завершён. Войдите снова.'); }
    let data = {};
    try { data = await response.json(); } catch { /* empty response */ }
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : data.detail?.reason;
      const messages = { 403: 'У этой учётной записи нет доступа к этому действию.', 404: 'Запись не найдена. Обновите список и попробуйте снова.', 409: 'Данные изменились или запрос ещё обрабатывается. Обновите состояние и повторите действие.', 422: 'Проверьте заполненные поля и повторите запрос.', 429: 'Очередь заполнена. Повторите запрос позже.', 502: 'Локальная модель не смогла подготовить ответ. Попробуйте ещё раз.', 503: 'Учебный сервис сейчас недоступен. Проверьте запуск базы и локальной модели.' };
      const error = new Error(messages[response.status] || detail || `Ошибка сервера (${response.status}).`); error.status = response.status; error.detail = detail; throw error;
    }
    return data;
  }
  function showLogin(message = '') {
    $('#workspace').hidden = true; $('#login-view').hidden = false;
    $('#login-error').hidden = !message; $('#login-error').textContent = message;
    $('#demo-login').hidden = true;
    $('#login-intro').textContent = 'Используйте выданную учётную запись учителя или ученика.';
    request('/auth/demo').then(data => {
      const available = data.enabled === true && ['teacher', 'student'].every(role => data.roles?.includes(role));
      $('#demo-login').hidden = !available;
      $('#login-intro').textContent = available ? 'Выберите демонстрацию или войдите в свою учётную запись.' : 'Используйте выданную учётную запись учителя или ученика.';
    }).catch(() => { $('#demo-login').hidden = true; });
  }
  function signOutLocal() {
    guidedContexts.clear(); sessionRecovery.clear(); sessionClosures.clear(); guidedViewRevision++; chatReturnAttempt = null;
    resetLearningArchive();
    state.token = null; state.account = null; state.learnerId = null; sessionStorage.removeItem(tokenKey);
    state.learners = []; state.dashboard = null; state.topics = []; state.task = null; state.proposal = null; state.revision = null; state.chatByLearner.clear(); state.nextStepsByLearner.clear(); state.learningSessionsByLearner.clear(); state.pendingMessages.clear(); state.chatTopicsByLearnerSubject.clear(); state.memoryCorrectionJobs.clear(); state.viewRevision++;
    $('#learner-list').replaceChildren(); $('#learner-count').textContent = '–'; $('#roster-empty').hidden = true; $('#profile-content').hidden = true; $('#loading-state').hidden = true; $('#error-state').hidden = true; $('#empty-profile').hidden = false;
    $('#account-name').textContent = ''; $('#role-label').textContent = ''; $('#learner-name').textContent = ''; $('#learner-avatar').textContent = ''; $('#profile-grade').textContent = ''; $('#profile-meta').textContent = ''; $('#current-goal').textContent = ''; $('#goal-subject').textContent = ''; $('#revision-label').textContent = ''; $('#zone-map').replaceChildren(); $('#history-list').replaceChildren(); $('#task-cards').replaceChildren(); $('#task-result').hidden = true; $('#proposal-result').replaceChildren(); $('#proposal-result').hidden = true;
    $('#topic-select').replaceChildren(new Option('Выберите тему', '')); $('#observation-topic').replaceChildren(new Option('Выберите тему', '')); $('#chat-subject').value = 'math'; $('#edit-goal').value = ''; $('#edit-interests').value = '';
    $('#chat-log').replaceChildren(el('div', 'chat-bubble assistant', greeting)); $('#chat-input').value = ''; $('#lesson-notes').value = ''; $('#observation-notes').value = ''; $('#observation-task').value = ''; $('#assessment-pair').value = ''; $('#observation-difficulty').value = 'medium';
    resetLearningAction();
    $('#email').value = ''; $('#password').value = '';
    $('#teacher-actions').hidden = true; $('#student-actions').hidden = true; $('#action-empty').hidden = false;
    $('#memory-section').hidden = true; $('#memory-history').replaceChildren(); $('#memory-status').textContent = ''; $('#next-step-result').replaceChildren(); $('#next-step-result').hidden = true; $('#next-step-status').textContent = ''; $('#memory-method').value = 'tail'; $('#request-next-step').disabled = false; renderLearningSession();
    if ($('#edit-dialog').open) $('#edit-dialog').close();
    $('#workspace').hidden = true; showLogin();
  }
  async function boot() {
    if (!state.token) return showLogin();
    try {
      const data = await request('/auth/me'); state.account = data.account;
      if (!state.account || !['teacher', 'student'].includes(state.account.role)) throw new Error('Для этой учётной записи рабочее место недоступно.');
      $('#login-view').hidden = true; $('#workspace').hidden = false;
      $('#account-name').textContent = displayName(state.account); $('#role-label').textContent = state.account.role === 'teacher' ? 'Учитель' : 'Ученик';
      $('#workspace-label').textContent = state.account.role === 'teacher' ? 'Рабочее место учителя' : 'Мой учебный профиль';
      $('#teacher-actions').hidden = state.account.role !== 'teacher'; $('#student-actions').hidden = state.account.role !== 'student';
      $('#memory-section').hidden = state.account.role !== 'teacher';
      $('#action-empty').hidden = state.account.role === 'student';
      $('#roster-title').textContent = state.account.role === 'teacher' ? 'Ученики' : 'Мой профиль';
      $('#learner-filter').hidden = state.account.role === 'student';
      $('#edit-profile').hidden = state.account.role !== 'teacher';
      await loadLearners();
      loadCatalog().catch(err => toast(errorText(err)));
    } catch (error) { showLogin(errorText(error)); }
  }
  async function loadLearners() {
    const list = $('#learner-list'); list.replaceChildren();
    list.append(el('div', 'state-message', 'Загружаем список…'));
    try {
      const data = await request('/learners');
      state.learners = Array.isArray(data.learners) ? data.learners : [];
      $('#learner-count').textContent = String(state.learners.length);
      renderRoster();
      if (!state.learners.length) { state.learnerId = null; showEmptyProfile(); return; }
      const desired = state.account.role === 'student' ? state.account.learner_id || state.learners[0].learner_id : state.learnerId;
      const first = state.learners.find(row => row.learner_id === desired) || state.learners[0];
      selectLearner(first.learner_id);
    } catch (error) { list.replaceChildren(); $('#roster-empty').hidden = false; $('#roster-empty').textContent = errorText(error); showError(error); }
  }
  function learnerRecord(row) { return row.profile || row; }
  function renderRoster() {
    const list = $('#learner-list'); list.replaceChildren(); $('#roster-empty').hidden = state.learners.length > 0;
    state.learners.forEach(row => {
      const card = learnerRecord(row), name = card.pseudonym || card.display_name || row.learner_id || 'Ученик';
      const button = el('button', `learner-item${row.learner_id === state.learnerId ? ' selected' : ''}`);
      button.type = 'button'; button.setAttribute('role', 'listitem'); button.dataset.id = row.learner_id;
      button.append(el('span', 'learner-avatar', initials(name)));
      const info = el('span'); info.append(el('strong', '', name), el('small', '', card.grade ? `${card.grade} класс` : 'Учебный профиль'));
      button.append(info); button.addEventListener('click', () => selectLearner(row.learner_id)); list.append(button);
    });
    filterRoster();
  }
  function filterRoster() {
    const query = ($('#learner-filter').value || '').trim().toLocaleLowerCase('ru'); let shown = 0;
    $$('.learner-item', $('#learner-list')).forEach(item => { const visible = item.textContent.toLocaleLowerCase('ru').includes(query); item.hidden = !visible; if (visible) shown++; });
    $('#roster-empty').hidden = !!query || state.learners.length > 0;
    if (query && !shown) { $('#roster-empty').hidden = false; $('#roster-empty').textContent = 'Совпадений нет. Попробуйте другое имя.'; }
    else if (!query && state.learners.length) $('#roster-empty').hidden = true;
  }
  function showEmptyProfile() { $('#profile-content').hidden = true; $('#loading-state').hidden = true; $('#error-state').hidden = true; $('#empty-profile').hidden = false; $('#action-empty').hidden = false; }
  function showError(error) { $('#profile-content').hidden = true; $('#loading-state').hidden = true; $('#empty-profile').hidden = true; $('#error-state').hidden = false; $('#error-text').textContent = errorText(error); }
  async function selectLearner(id) {
    const viewRevision = ++state.viewRevision;
    guidedViewRevision++; chatReturnAttempt = null;
    if (state.learnerId !== id) resetLearningAction();
    state.learnerId = id; state.dashboard = null; state.task = null; state.proposal = null; renderRoster();
    resetLearningArchive();
    if (state.account?.role === 'student') { renderChatHistory(id); restoreLearningSession(); }
    renderNextStep(); $('#memory-history').replaceChildren(); $('#memory-status').textContent = '';
    $('#profile-content').hidden = true; $('#error-state').hidden = true; $('#empty-profile').hidden = true; $('#loading-state').hidden = false; $('#action-empty').hidden = true;
    $('#task-result').hidden = true; $('#proposal-result').hidden = true;
    try {
      const data = await request(`/learners/${encodeURIComponent(id)}/dashboard`);
      if (viewRevision !== state.viewRevision) return;
      state.dashboard = data; state.revision = data.revision ?? data.profile?.revision ?? 1;
      renderDashboard(data); $('#loading-state').hidden = true; $('#profile-content').hidden = false;
      $('#action-empty').hidden = true;
      fillTopicOptions();
      if (state.account?.role === 'student') { renderLearningSession(); refreshGuided(); }
      renderNextStep();
      if (state.account?.role === 'teacher') { loadMemory(); const pending = state.nextStepsByLearner.get(id); if (pending?.job_id && !pending.result && !pending.finished) pollNextStep(id, pending.job_id, viewRevision); }
    } catch (error) { if (viewRevision !== state.viewRevision) return; showError(error); $('#action-empty').hidden = true; }
  }
  const subjectTitles = { math: 'Математика', python: 'Python', programming: 'Программирование', scratch: 'Scratch', general: 'Общее занятие' };
  const topicTitle = topic => topic?.topic_title || topic?.title || topic?.title_ru || topic?.name || topic?.topic || subjectTitles[topic?.topic_id || topic?.id] || topic?.topic_id || topic?.id || 'Тема не указана';
  function normalizeTopics(data) { return Array.isArray(data.topics) ? data.topics : Array.isArray(data) ? data : []; }
  function topicRows(dashboard) { return dashboard?.zpd?.topics || dashboard?.zone_report?.topics || dashboard?.profile?.zpd?.topics || []; }
  function renderDashboard(d) {
    const p = d.profile || {}, rag = p.rag_context || {}, goal = rag.current_goal || p.current_goal || {};
    const name = p.pseudonym || p.display_name || state.learners.find(x => x.learner_id === state.learnerId)?.profile?.pseudonym || 'Ученик';
    $('#learner-name').textContent = name; $('#learner-avatar').textContent = initials(name);
    $('#profile-grade').textContent = p.grade ? `Учебный профиль · ${p.grade} класс` : 'Учебный профиль';
    const meta = [p.language ? `Язык: ${languageLabel(p.language)}` : '', Array.isArray(p.interests) && p.interests.length ? `Интересы: ${p.interests.slice(0, 3).join(', ')}` : ''].filter(Boolean).join(' · ');
    $('#profile-meta').textContent = meta;
    $('#current-goal').textContent = p.goal?.text || goal.goal || goal.description || (state.account.role === 'student' ? 'Цель ещё не указана' : 'Добавьте цель в профиле или заметках к занятию');
    const goalTopic = goal.topic_id ? topicById(goal.topic_id) : null;
    $('#goal-subject').textContent = p.goal?.topic || (goalTopic ? topicTitle(goalTopic) : goal.topic_id ? topicTitle({ topic_id: goal.topic_id }) : '');
    $('#revision-label').textContent = state.account.role === 'teacher' && d.revision ? `Версия профиля ${d.revision}` : '';
    renderZones(topicRows(d)); renderHistory(d.history?.observations || d.observations || []);
    if (state.account.role === 'teacher') { $('#edit-goal').value = goal.goal || ''; $('#edit-interests').value = (p.interests || []).join(', '); }
  }
  function topicById(id) { return state.topics.find(t => (t.topic_id || t.id) === id); }
  function zoneInfo(row) {
    const evidence = row.evidence || {}, status = row.evidence_status || 'insufficient';
    const studentZone = state.account.role === 'student';
    const progress = String(row.progress || '');
    const established = studentZone
      ? !['Продолжаем наблюдать', 'Нужно проверить'].includes(progress)
      : status === 'confirmed' && evidence.sufficient === true;
    let zone = studentZone
      ? (progress === 'Получается самостоятельно' ? 'mastered' : progress === 'Учится с поддержкой' ? 'zpd' : progress === 'Нужна подготовительная опора' ? 'outside' : 'unknown')
      : established ? (row.confirmed_zone || 'unknown') : 'unknown';
    const mapping = {
      mastered: ['Самостоятельно', 'В сопоставимых заданиях справляется без подсказок', '#3e796b'],
      independent: ['Самостоятельно', 'В сопоставимых заданиях справляется без подсказок', '#3e796b'],
      zpd: ['С поддержкой', 'Получается с доступной опорой взрослого', '#cf7430'],
      assisted: ['С поддержкой', 'Получается с доступной опорой взрослого', '#cf7430'],
      outside: ['Пока не получилось с совместной помощью', 'В последних сопоставимых попытках не получилось самостоятельно и при совместном решении со взрослым. Это описывает только эти попытки.', '#986c64'],
      unknown: ['Нужно проверить', status === 'stale' ? 'Наблюдения устарели – их стоит повторить' : status === 'inconclusive' ? 'Результаты пока расходятся' : 'Пока недостаточно проверок', '#728d9a'],
      observe: ['Нужно проверить', 'Пока недостаточно проверок', '#728d9a'],
      not_started: ['Нужно проверить', 'Пока нет наблюдений по этой теме', '#728d9a'],
      not_yet: ['Нужно проверить', 'Пока нет наблюдений по этой теме', '#728d9a']
    };
    const selection = mapping[zone] || mapping.observe;
    const successful = Number(evidence.independent_successes || evidence.independent_success_count || 0) + Number(evidence.assisted_successes || evidence.assisted_success_count || 0);
    const attempts = Number(evidence.independent_attempts || evidence.independent_attempt_count || 0) + Number(evidence.assisted_attempts || evidence.assisted_attempt_count || 0);
    const note = row.confirmed_reason || (established ? `${successful} успешных попыток из ${attempts}` : selection[1]);
    return { ...selection, title: selection[0], note, count: attempts ? `${attempts} наблюд.` : 'нет данных', established, color: selection[2] };
  }
  function renderZones(rows) {
    const map = $('#zone-map'); map.replaceChildren();
    const buckets = [
      { key: 'independent', title: 'Самостоятельно', color: '#3e796b', tests: r => zoneInfo(r).title === 'Самостоятельно' },
      { key: 'support', title: 'С поддержкой', color: '#cf7430', tests: r => zoneInfo(r).title === 'С поддержкой' },
      { key: 'outside', title: 'Пока не получилось с совместной помощью', color: '#986c64', tests: r => zoneInfo(r).title === 'Пока не получилось с совместной помощью' },
      { key: 'unknown', title: 'Нужно проверить', color: '#728d9a', tests: r => zoneInfo(r).title === 'Нужно проверить' }
    ];
    for (const bucket of buckets) {
      const matched = rows.filter(bucket.tests), section = el('section', 'zone'); section.style.setProperty('--zone-color', bucket.color);
      const header = el('div', 'zone-head'); header.append(el('h3', '', bucket.title), el('span', 'zone-count', `${matched.length} тем`)); section.append(header);
      if (!matched.length) section.append(el('p', 'zone-empty', 'Тем пока нет'));
      const pills = el('div', 'topic-pills');
      matched.forEach(row => {
        const id = row.topic_id || row.id, t = topicById(id) || row, button = el('button', 'topic-pill', topicTitle(t)); button.type = 'button'; button.title = zoneInfo(row).note;
        if (state.account.role === 'teacher') button.addEventListener('click', () => { $('#topic-select').value = id || ''; $('#observation-topic').value = id || ''; loadMemory(); document.querySelector('[data-panel="action-panel"]').click(); });
        else { button.disabled = true; button.setAttribute('aria-label', `${topicTitle(t)}. ${zoneInfo(row).title}`); }
        pills.append(button);
      });
      section.append(pills);
      map.append(section);
    }
    if (!rows.length) map.append(el('p', 'zone-empty', state.account.role === 'student' ? 'Темы появятся здесь после работы с ними.' : 'Для этого профиля пока нет данных по темам. Добавьте наблюдение или заметку к занятию.'));
  }
  function safeDate(raw) { if (!raw) return 'Недавно'; const d = new Date(raw); return Number.isNaN(d.getTime()) ? String(raw) : new Intl.DateTimeFormat('ru', { day: 'numeric', month: 'short' }).format(d); }
  function renderHistory(rows) {
    const root = $('#history-list'); root.replaceChildren();
    if (!Array.isArray(rows) || !rows.length) { root.append(el('p', 'zone-empty', 'Наблюдений пока нет. Когда появятся первые попытки, здесь будет видна история работы.')); return; }
    rows.slice(0, 8).forEach(row => {
      const item = el('article', 'history-item'), date = el('time', 'history-date', safeDate(row.observed_on || row.created_at || row.date));
      const copy = el('div', 'history-copy'), topic = topicById(row.topic_id), title = topicTitle(topic || row);
      const solved = row.solved === true || row.outcome === 'solved' || row.success === true || ['Выполнено самостоятельно', 'Выполнено с опорой'].includes(row.outcome);
      const support = row.support_level || row.support || (row.outcome === 'Выполнено с опорой' ? 'assisted' : 'independent');
      const supportLabel = ({ independent: 'самостоятельно', hint_question: 'после наводящего вопроса', visual: 'с наглядной опорой', joint: 'при совместной работе', assisted: 'с опорой', not_solved: 'попытка не завершена' })[support] || 'с опорой';
      const outcome = state.account.role === 'student' && row.outcome ? row.outcome : solved ? `решил ${supportLabel}` : supportLabel;
      const heading = el('strong'); heading.append(el('i', `history-mark${solved ? (support === 'independent' ? '' : ' supported') : ' no'}`), document.createTextNode(`${title} – ${outcome}`));
      copy.append(heading); if (row.notes) copy.append(el('p', '', row.notes)); else if (row.task_title || row.task_id) copy.append(el('p', '', row.task_title || row.task_id));
      item.append(date, copy); root.append(item);
    });
  }
  async function loadCatalog() {
    const data = await request('/catalog'); state.topics = normalizeTopics(data); fillTopicOptions();
    if (state.dashboard) { renderDashboard(state.dashboard); renderNextStep(); }
    if (state.account?.role === 'student') renderLearningSession();
  }
  function fillTopicOptions() {
    for (const select of [$('#topic-select'), $('#observation-topic')]) {
      const previous = select.value; select.replaceChildren();
      const first = document.createElement('option'); first.value = ''; first.textContent = select.id === 'topic-select' ? 'Выберите тему' : 'Выберите тему'; select.append(first);
      const goalId = state.dashboard?.profile?.rag_context?.current_goal?.topic_id;
      const candidates = [...state.topics];
      if (goalId && !candidates.some(t => (t.topic_id || t.id) === goalId)) candidates.unshift({ topic_id: goalId, title: goalId });
      candidates.forEach(t => { const id = t.topic_id || t.id; const option = document.createElement('option'); option.value = id; option.textContent = `${topicTitle(t)}${t.grade ? ` · ${t.grade} класс` : ''}`; select.append(option); });
      if ([...select.options].some(o => o.value === previous)) select.value = previous;
      else if (goalId && select.id === 'topic-select') select.value = goalId;
    }
  }
  async function busy(button, fn) { if (button.disabled) return; const prior = button.textContent; button.disabled = true; button.dataset.prior = prior; try { await fn(); } catch (error) { toast(errorText(error)); } finally { button.disabled = false; button.textContent = button.dataset.prior || prior; delete button.dataset.prior; } }
  function toast(message) { const node = $('#toast'); node.textContent = message; node.classList.add('visible'); clearTimeout(toastTimer); toastTimer = setTimeout(() => node.classList.remove('visible'), 3400); }
  function renderTask(data) {
    state.task = data; const result = data.card || {}; $('#task-result').hidden = false; const root = $('#task-cards'); root.replaceChildren();
    const tasks = Array.isArray(result.tasks) ? result.tasks : Array.isArray(result.items) ? result.items : [];
    if (!tasks.length && data.text_for_child) { const card = el('article', 'task-card'); card.append(el('p', '', data.text_for_child)); root.append(card); }
    tasks.forEach((task, index) => {
      const article = el('article', 'task-card'); article.append(el('span', 'task-number', `Задание ${index + 1}`));
      article.append(el('p', '', task.prompt || task.content || task.task || task.question || 'Задание подготовлено'));
      if (task.hint) article.append(el('p', 'task-source', `Подсказка: ${task.hint}`));
      if ($('#show-answers').checked && (task.answer || task.solution || task.expected_answer)) article.append(el('p', 'task-answer', `Ответ: ${task.answer || task.solution || task.expected_answer}`));
      if (task.source || task.source_file || task.citation) article.append(el('p', 'task-source', `Источник: ${task.source || task.source_file || task.citation}`));
      root.append(article);
    });
    if (data.review_required) root.append(el('p', 'review-note', 'Материал создан как черновик. Учитель проверяет его перед использованием.'));
  }
  function sourceDetails(sources, title = 'Источники', options = {}) {
    const details = el('details', 'memory-sources'); details.append(el('summary', '', `${title} (${sources.length})`));
    sources.forEach(source => {
      const row = el('article', 'memory-source');
      row.append(el('strong', '', source.source_id || source.turn_id || source.id || 'Источник занятия'));
      const sourceRole = source.role || source.author_role;
      const role = ({ user: 'Ученик', student: 'Ученик', teacher: 'Учитель', assistant: 'Помощник' })[sourceRole] || sourceRole || '';
      row.append(el('small', '', [role, source.topic_id || source.topic_title ? topicTitle(source) : '', safeDate(source.observed_on || source.created_at)].filter(Boolean).join(' · ')));
      if (options.showDialogueContext && source.cited === false) row.append(el('small', '', 'Связанная запись для проверки исправления; в ссылках черновика не указана.'));
      if (options.showDialogueContext && Array.isArray(source.corrected_by) && source.corrected_by.length) row.append(el('p', 'review-note', `Запись исправлена: ${source.corrected_by.join(', ')}. Сверьте её с исправлениями.`));
      const dialogueContext = options.showDialogueContext && source.dialogue_context;
      if (dialogueContext?.role === 'assistant' && dialogueContext.usage === 'context_only' && typeof dialogueContext.text === 'string') {
        const context = el('div', 'recommendation-dialogue-context');
        context.append(el('small', '', 'Реплика помощника перед ответом ученика'), el('blockquote', 'memory-text', dialogueContext.text), el('small', '', 'Контекст вопроса. Слова помощника не подтверждают результат ученика.'));
        if (dialogueContext.truncated) context.append(el('small', '', 'Показан только конец реплики помощника.'));
        row.append(context);
      }
      if (options.showDialogueContext) row.append(el('small', '', 'Буквальная реплика источника'));
      row.append(el(options.showDialogueContext ? 'blockquote' : 'p', 'memory-text', source.text || source.content || source.quote || 'Текст источника недоступен.'));
      const sessionEndContext = source.session_end_context;
      if (sessionEndContext?.role === 'assistant' && sessionEndContext.usage === 'context_only' && typeof sessionEndContext.text === 'string' && sessionEndContext.text) {
        const context = el('div', 'memory-session-end-context');
        context.append(el('small', '', 'Последняя реплика помощника в занятии'), el('blockquote', 'memory-text', sessionEndContext.text), el('small', '', 'Только контекст разговора. Результат ученика по этой реплике не установлен.'));
        row.append(context);
      }
      if (source.session_end_context_omitted) row.append(el('p', 'review-note', 'Последняя реплика помощника слишком длинная и не включена в память.'));
      if (source.corrects_source_id) row.append(el('small', '', `Исправляет источник ${source.corrects_source_id}. Исходная запись сохранена.`));
      if (options.onCorrection && source.source_id && ['student', 'teacher'].includes(sourceRole)) options.onCorrection(row, source);
      details.append(row);
    });
    return details;
  }
  function recordLabel(record) { return record.claim || record.text || record.summary || record.observation || record.content || 'Запись о занятии'; }
  function appendMemoryRecords(root, records) {
    records.forEach(record => {
      const row = el('article', 'memory-record'); row.append(el('p', '', recordLabel(record)));
      if (record.corrects_record_id) row.append(el('small', '', `Исправляет запись ${record.corrects_record_id}. Исходное наблюдение сохранено.`));
      if (record.support_level === 'unknown' || record.assistance === 'unknown') row.append(el('small', '', 'Условия помощи неизвестны.'));
      const ids = record.evidence_turn_ids || record.source_ids || (record.source_id ? [record.source_id] : []);
      if (Array.isArray(ids) && ids.length) row.append(el('small', '', `Основания: ${ids.join(', ')}`));
      if (record.quote) row.append(el('blockquote', '', record.quote)); root.append(row);
    });
  }
  async function loadMemory() {
    const learnerId = state.learnerId, viewRevision = state.viewRevision, topic = $('#topic-select').value, sessionToken = state.token;
    if (!learnerId || state.account?.role !== 'teacher') return;
    const memoryViewRevision = ++state.memoryViewRevision;
    const currentSelection = () => sessionToken === state.token && learnerId === state.learnerId && viewRevision === state.viewRevision && topic === $('#topic-select').value && state.account?.role === 'teacher';
    const currentView = () => currentSelection() && memoryViewRevision === state.memoryViewRevision;
    $('#memory-status').textContent = 'Загружаем источники и версии памяти…';
    try {
      const data = await request(`/learners/${encodeURIComponent(learnerId)}/memory${topic ? `?topic_id=${encodeURIComponent(topic)}` : ''}`);
      if (!currentView()) return;
      const root = $('#memory-history'); root.replaceChildren();
      const sources = Array.isArray(data.sources) ? data.sources : [], versions = Array.isArray(data.versions) ? data.versions : [];
      $('#memory-status').textContent = `Версия профиля ${data.revision ?? '–'}. ${data.human_reviewed ? 'Содержание просмотрено преподавателем.' : 'Содержание памяти требует просмотра преподавателем.'}`;
      const correctionStatus = status => ({
        queued: 'Исправление сохранено. Обновление памяти в очереди.',
        pending: 'Исправление сохранено. Обновление памяти в очереди.',
        running: 'Исправление сохранено. Память обновляется.',
        completed: 'Исправление сохранено. Обновление памяти завершено. Проверьте обновлённую версию памяти.',
        succeeded: 'Исправление сохранено. Обновление памяти завершено. Проверьте обновлённую версию памяти.',
        failed: 'Исправление сохранено. Обновить память не удалось. Автоматического повторного запуска не будет.',
        cancelled: 'Исправление сохранено. Обновление памяти отменено. Автоматического повторного запуска не будет.',
        missing_job: 'Исправление сохранено. Сервер не вернул идентификатор обновления памяти. Обновите список позже.',
        unknown_outcome: 'Исправление сохранено. Исход обновления памяти неизвестен. Автоматического повторного запуска не будет.',
        outcome_unknown: 'Исправление сохранено. Исход обновления памяти неизвестен. Автоматического повторного запуска не будет.',
        unknown: 'Исправление сохранено. Исход обновления памяти неизвестен. Автоматического повторного запуска не будет.',
      })[status] || 'Исправление сохранено. Состояние обновления памяти ещё не проверено.';
      state.memoryCorrectionJobs.forEach(job => {
        if (job.learnerId !== learnerId || (topic && job.topic !== topic)) return;
        const panel = el('article', 'memory-record memory-correction-job');
        panel.append(el('small', '', `Исправление источника ${job.sourceId}`));
        const status = el('p', 'review-note', job.statusText || correctionStatus(job.status)); status.setAttribute('role', 'status'); panel.append(status);
        if (job.job_id) {
          const check = el('button', 'button secondary', 'Проверить обновление памяти'); check.type = 'button';
          check.addEventListener('click', () => busy(check, async () => {
            if (!currentView()) return;
            status.textContent = 'Исправление сохранено. Проверяем состояние обновления памяти…';
            try {
              const result = await request(`/memory/jobs/${encodeURIComponent(job.job_id)}`);
              if (!currentView()) return;
              job.status = result.status || 'unknown'; job.statusText = correctionStatus(job.status); status.textContent = job.statusText;
              if (['completed', 'succeeded'].includes(job.status)) {
                const refreshed = await loadMemory();
                if (currentSelection() && refreshed?.generation === state.memoryViewRevision) $('#memory-status').textContent = `${job.statusText}${refreshed.rendered ? '' : ' Не удалось обновить список памяти. Нажмите «Обновить».'}`;
              }
            } catch (error) {
              if (currentView()) { job.statusText = `Исправление сохранено. Проверить обновление памяти не удалось. ${errorText(error)}`; status.textContent = job.statusText; }
            }
          })); panel.append(check);
        }
        root.append(panel);
      });
      if (!sources.length && !versions.length) root.append(el('p', 'zone-empty', 'Завершённых занятий пока нет. Источники появятся после сохранения занятия или диалога.'));
      versions.forEach(version => {
        const details = el('details', 'memory-version');
        const names = { tail: 'Последние занятия', profile_only: 'Профиль', summary: 'Сводка', structured: 'Структурированная память' };
        details.append(el('summary', '', `${topicTitle(version)} · ${names[version.method] || 'Память'} · ${safeDate(version.created_at)}${version.stale ? ' · требует обновления' : ''}${version.approved ? ' · подтверждена' : ' · черновик'}`));
        const memory = version.state || version.memory || version.payload || version;
        const records = memory.records || version.records || [];
        if (Array.isArray(records)) appendMemoryRecords(details, records);
        const text = memory.summary || memory.text || version.context?.text || version.context_text;
        if (typeof text === 'string' && text) details.append(el('p', 'memory-text', text));
        if (Array.isArray(version.context?.selection_events) && version.context.selection_events.some(event => event.event === 'source_correction_group_skipped')) details.append(el('p', 'review-note', 'Исходная запись и исправления не поместились в выбранную память. Эта связка не будет передана помощнику.'));
        if (Array.isArray(version.context?.selection_events) && version.context.selection_events.some(event => event?.event === 'session_end_context_group_skipped')) details.append(el('p', 'review-note', 'Контекст последней реплики помощника не поместился в выбранную память и не будет передан помощнику.'));
        if (Array.isArray(version.context?.selection_events) && version.context.selection_events.some(event => event?.event === 'session_end_context_omitted')) details.append(el('p', 'review-note', 'Последняя реплика помощника слишком длинная и не включена в память. Контекст этой реплики недоступен.'));
        if (version.context?.interest_extraction?.status === 'failed') details.append(el('p', 'review-note', 'Учебная память готова. Интересы из разговора извлечь не удалось.'));
        const canApprove = version.version_id && version.method === 'tail' && !version.approved && !version.stale;
        const candidates = Array.isArray(version.context?.interest_candidates) ? version.context.interest_candidates : [];
        const interestChoices = [];
        if (candidates.length) {
          const operationLabels = { add: 'Добавить интерес', remove: 'Убрать интерес', avoid: 'Не предлагать этот сюжет', allow: 'Снова разрешить этот сюжет' };
          details.append(el('h4', '', 'Интересы из разговора'));
          details.append(el('p', 'review-note', canApprove ? 'В карточку попадут только отмеченные интересы. После подтверждения выбор сохраняется вместе с памятью.' : 'Предложения к этой версии памяти. Они добавляются в карточку только при отдельном выборе преподавателя.'));
          candidates.forEach(candidate => {
            if (!candidate.candidate_id || !candidate.interest || !Object.hasOwn(operationLabels, candidate.operation)) return;
            const row = el('article', 'memory-record');
            const label = `${operationLabels[candidate.operation]}: ${candidate.interest}`;
            if (canApprove) {
              const choice = el('input'); choice.type = 'checkbox'; choice.value = candidate.candidate_id;
              const check = el('label', 'check-line'); check.append(choice, el('span', '', label)); row.append(check); interestChoices.push(choice);
            } else row.append(el('p', '', `Предложение – ${operationLabels[candidate.operation].toLocaleLowerCase('ru')}: ${candidate.interest}`));
            if (candidate.quote) row.append(el('blockquote', '', candidate.quote));
            if (candidate.source_id) row.append(el('small', '', `Источник: ${candidate.source_id}`));
            details.append(row);
          });
        }
        if (canApprove) {
          const approve = el('button', 'button secondary', 'Подтвердить память для помощника'); approve.type = 'button';
          details.append(el('p', 'review-note', 'Проверьте записи и исходные реплики. Подтверждённая память станет доступна помощнику ученика.'));
          approve.addEventListener('click', () => busy(approve, async () => {
            if (viewRevision !== state.viewRevision || topic !== $('#topic-select').value) return;
            const interestCandidateIds = interestChoices.filter(choice => choice.checked).map(choice => choice.value);
            interestChoices.forEach(choice => { choice.disabled = true; });
            try {
              await request(`/learners/${encodeURIComponent(learnerId)}/memory/versions/${encodeURIComponent(version.version_id)}/confirm`, { method: 'POST', body: { expected_revision: data.revision, interest_candidate_ids: interestCandidateIds } });
              if (viewRevision !== state.viewRevision) return;
              const dashboard = await request(`/learners/${encodeURIComponent(learnerId)}/dashboard`);
              if (viewRevision !== state.viewRevision) return;
              state.dashboard = dashboard; state.revision = dashboard.revision ?? dashboard.profile?.revision ?? state.revision;
              renderDashboard(dashboard); renderNextStep();
              await loadMemory();
              if (viewRevision === state.viewRevision) toast('Память подтверждена');
            } catch (error) {
              if (viewRevision === state.viewRevision) { interestChoices.forEach(choice => { choice.disabled = false; }); throw error; }
            }
          })); details.append(approve);
        }
        root.append(details);
      });
      if (sources.length) root.append(sourceDetails(sources, 'Исходные реплики и заметки', { onCorrection: Number.isInteger(data.revision) && Number.isInteger(data.source_revision) ? (row, source) => {
        const open = el('button', 'button secondary', 'Исправить запись'); open.type = 'button';
        const form = el('form', 'work-form compact-form'); form.hidden = true;
        const input = el('textarea'); input.setAttribute('aria-label', 'Исправление записи'); input.maxLength = 2000; input.required = true; input.rows = 4;
        const save = el('button', 'button secondary', 'Сохранить исправление'); save.type = 'submit';
        const status = el('p', 'review-note'); status.setAttribute('role', 'status');
        form.append(el('p', 'review-note', 'Исходная запись сохранится. Это исправление сведений о прошлом занятии, а не новый результат или прогресс ребёнка.'), input, save, status);
        let pending = null, submitting = false;
        open.addEventListener('click', () => { form.hidden = !form.hidden; if (!form.hidden) input.focus(); });
        input.addEventListener('input', () => { if (pending && pending.text !== input.value) pending = null; });
        form.addEventListener('submit', async event => {
          event.preventDefault();
          if (submitting || !currentView() || !input.value.trim()) return;
          if (!pending || pending.text !== input.value) pending = { text: input.value, client_correction_id: crypto.randomUUID() };
          const correction = pending;
          submitting = true; save.disabled = true; input.disabled = true; open.disabled = true; status.textContent = 'Сохраняем исправление…';
          try {
            const result = await request(`/learners/${encodeURIComponent(learnerId)}/memory/sources/${encodeURIComponent(source.source_id)}/corrections`, { method: 'POST', body: { text: correction.text, expected_revision: data.revision, expected_source_revision: data.source_revision, client_correction_id: correction.client_correction_id } });
            if (sessionToken !== state.token || state.account?.role !== 'teacher') return;
            const job = { job_id: result.job_id, learnerId, topic: result.topic_id || source.topic_id || topic, sourceId: source.source_id, status: result.job_id ? 'not_checked' : 'missing_job' };
            state.memoryCorrectionJobs.set(result.job_id || correction.client_correction_id, job);
            if (!currentView()) return;
            const refreshed = await loadMemory();
            if (currentSelection() && refreshed?.generation === state.memoryViewRevision) $('#memory-status').textContent = `${correctionStatus(job.status)}${refreshed.rendered ? '' : ' Не удалось обновить список памяти. Нажмите «Обновить».'}`;
          } catch (error) {
            if (currentView()) status.textContent = `${errorText(error)} Черновик исправления сохранён в форме.${error.status ? '' : ' Исход запроса может быть неизвестен. Повтор без изменения текста использует тот же идентификатор.'}`;
          } finally {
            submitting = false; save.disabled = false; input.disabled = false; open.disabled = false;
          }
        });
        row.append(open, form);
      } : null }));
      return { rendered: true, generation: memoryViewRevision };
    } catch (error) { if (currentView()) $('#memory-status').textContent = errorText(error); return { rendered: false, generation: memoryViewRevision }; }
  }
  function renderNextStep() {
    const current = state.nextStepsByLearner.get(state.learnerId), root = $('#next-step-result'); root.replaceChildren(); root.hidden = !current?.result;
    $('#next-step-status').textContent = current?.statusText || '';
    $('#request-next-step').disabled = !!current?.submitting || (!!current?.job_id && !current.finished);
    $('#request-next-step').textContent = current?.job_id && !current.finished ? 'Проверить состояние' : 'Предложить следующий шаг';
    if (!current?.result) return;
    const data = current.result;
    const actionTitles = { enough_data_next_task: 'Предложить следующую задачу', independent_check: 'Провести самостоятельную проверку', continue_work: 'Продолжить работу', clarify_missing_observation: 'Уточнить недостающее наблюдение' };
    const actionTitle = Object.hasOwn(actionTitles, data.action) ? actionTitles[data.action] : data.action;
    root.append(el('h3', '', 'Предложенное действие'), el('p', 'review-note', topicTitle(topicById(data.topic_id) || { topic_id: data.topic_id })), el('p', 'next-step-action', actionTitle || 'Действие не указано.'), el('strong', '', 'Черновик обоснования'), el('p', '', data.reason || 'Обоснование не указано.'));
    if (Array.isArray(data.unknowns) && data.unknowns.length) { root.append(el('strong', '', 'Что нужно уточнить')); const list = el('ul'); data.unknowns.forEach(item => list.append(el('li', '', item))); root.append(list); }
    else root.append(el('p', 'review-note recommendation-empty-unknowns', 'В черновике неизвестные условия не перечислены. Это не подтверждает самостоятельность или освоение.'));
    const review = data.evidence_review;
    if (review?.version === 'recommendation-source-review-v1') {
      if (Array.isArray(review.source_unknowns) && review.source_unknowns.length) {
        const notes = el('section', 'recommendation-source-unknowns'); notes.append(el('strong', '', 'Неопределённость, отмеченная в источниках'));
        review.source_unknowns.forEach(item => {
          const row = el('article', 'memory-record');
          row.append(el('small', '', `${item.condition === 'support' ? 'Условия помощи' : 'Самостоятельность'} · ${item.source_id}`), el('blockquote', '', item.quote));
          if (item.cited === false) row.append(el('small', '', 'Источник не указан в ссылках черновика.'));
          notes.append(row);
        }); root.append(notes);
      }
      if (Array.isArray(review.sources) && review.sources.length) {
        const evidence = sourceDetails(review.sources, 'Реплики для проверки обоснования', { showDialogueContext: true }); evidence.classList.add('recommendation-evidence'); root.append(evidence);
      }
      root.append(el('p', 'review-note', 'Сверьте обоснование с репликами. Допустимая ссылка сама по себе не подтверждает верность вывода.'));
    }
    if (data.review_required) root.append(el('p', 'review-note', 'Требуется просмотр преподавателем.'));
    if (data.stale) root.append(el('p', 'review-note', 'Использована последняя готовая память. Есть ещё не обработанные занятия.'));
    if (Array.isArray(data.sources) && data.sources.length) root.append(sourceDetails(data.sources));
    const use = el('button', 'button secondary wide', 'Создать задание по рекомендации'); use.type = 'button';
    use.disabled = !data.next_step_id || (data.revision != null && data.revision !== state.revision);
    if (use.disabled && data.next_step_id) root.append(el('p', 'review-note', 'Профиль изменился. Подготовьте рекомендацию заново.'));
    use.addEventListener('click', () => {
      if (state.nextStepsByLearner.get(state.learnerId)?.result !== data) return;
      const select = $('#topic-select');
      if (![...select.options].some(option => option.value === data.topic_id)) select.add(new Option(data.topic_id, data.topic_id));
      select.value = data.topic_id; current.useForTask = true;
      $('#task-form').requestSubmit();
    }); root.append(use);
  }
  async function pollNextStep(learnerId, jobId, viewRevision) {
    const current = state.nextStepsByLearner.get(learnerId);
    if (!current || current.pollingRevision === viewRevision) return;
    current.pollingRevision = viewRevision;
    try {
      for (let attempt = 0; attempt < 150; attempt++) {
        if (viewRevision !== state.viewRevision || !state.token) return;
        const job = await request(`/memory/jobs/${encodeURIComponent(jobId)}`);
        if (viewRevision !== state.viewRevision) return;
        if (['succeeded', 'completed'].includes(job.status)) { current.result = job.result; current.finished = true; current.statusText = 'Рекомендация готова к просмотру.'; renderNextStep(); loadMemory(); return; }
        if (['failed', 'unknown_outcome', 'outcome_unknown', 'cancelled', 'unknown'].includes(job.status)) { current.finished = true; current.statusText = `Рекомендация не подготовлена. ${typeof job.error === 'string' ? job.error : job.error?.message || 'Проверьте состояние сервиса.'}`; renderNextStep(); return; }
        current.statusText = job.status === 'running' ? 'Готовим рекомендацию…' : 'Ожидаем свободного места для генерации…'; renderNextStep();
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      current.statusText = 'Задание продолжает выполняться. Нажмите «Проверить состояние».'; renderNextStep();
    } catch (error) { if (viewRevision === state.viewRevision) { current.statusText = `${errorText(error)} Задание сохранено. Проверьте состояние позже.`; renderNextStep(); } }
    finally {
      if (current.pollingRevision === viewRevision) delete current.pollingRevision;
      if (viewRevision === state.viewRevision && !current.finished) { $('#request-next-step').disabled = false; $('#request-next-step').textContent = 'Проверить состояние'; }
    }
  }
  function appendProposalMemory(root, data) {
    let count = 0;
    const groups = Array.isArray(data.memory_records) ? [{ memory_records: data.memory_records }] : [...(data.cards || []), ...(data.updates || [])];
    if (!groups.some(group => group.memory_records?.length || group.proposed_memory_records?.length)) Object.entries(data.metadata?.memory_sources || {}).forEach(([learner_id, source]) => groups.push({ learner_id, memory_records: [{ ...source, support_level: 'unknown' }] }));
    groups.forEach(group => {
      const records = group.memory_records || group.proposed_memory_records || [];
      if (!Array.isArray(records) || !records.length) return;
      if (!count) root.append(el('h4', '', 'Записи памяти для подтверждения'));
      appendMemoryRecords(root, records); count += records.length;
    });
    if (count) root.append(el('p', 'review-note', 'Подтверждение сохраняет эти записи вместе с изменениями профиля. Выводы модели о навыках требуют отдельной проверки.'));
    return count;
  }
  function addProposalSummary(data) {
    const box = $('#proposal-result'); box.replaceChildren(); box.hidden = false;
    box.append(el('h3', '', 'Предложение к профилю'));
    const updates = data.updates || [];
    const memoryCount = appendProposalMemory(box, data);
    if (!updates.length && !memoryCount) { box.append(el('p', '', 'Из заметки не удалось определить изменение профиля. Уточните тему и наблюдаемый результат.')); return; }
    updates.forEach(update => {
      const change = el('div', 'proposal-change'); change.append(el('p', '', update.summary || update.changes?.join(', ') || 'Профиль будет обновлён после проверки.'));
      (update.warnings || []).forEach(warning => change.append(el('p', '', `Для проверки: ${warning}`)));
      box.append(change);
    });
    if (data.proposal_id) {
      state.proposal = data.proposal_id;
      const controls = el('div', 'proposal-actions'), review = el('button', 'button secondary', 'Посмотреть профиль'), confirm = el('button', 'button primary', 'Подтвердить'); review.type = confirm.type = 'button';
      review.addEventListener('click', () => busy(review, async () => { const proposal = await request(`/lessons/proposals/${encodeURIComponent(state.proposal)}`); renderProposal(proposal); }));
      confirm.addEventListener('click', () => busy(confirm, async () => { await request(`/lessons/proposals/${encodeURIComponent(state.proposal)}/confirm`, { method: 'POST' }); toast('Изменение подтверждено'); $('#proposal-result').hidden = true; await selectLearner(state.learnerId); }));
      controls.append(review, confirm); box.append(controls);
    }
  }
  function renderProposal(data) {
    const box = $('#proposal-result'); box.replaceChildren(); box.hidden = false; box.append(el('h3', '', 'Проверьте профиль перед подтверждением'));
    let foundChanges = false;
    (data.cards || []).forEach(row => {
      const profile = row.proposed_card || {}, before = state.dashboard?.profile || {}, goal = profile.rag_context?.current_goal;
      const oldGoal = before.rag_context?.current_goal?.goal || 'не указана', nextGoal = goal?.goal || 'не указана';
      box.append(el('p', '', `Профиль ${row.learner_id || state.learnerId} · версия ${row.base_revision ?? '–'}`));
      const compare = (label, oldValue, nextValue) => {
        const pretty = value => Array.isArray(value) ? value.map(id => topicTitle(topicById(id) || { topic_id: id })).join(', ') || 'нет' : String(value || 'нет');
        if (JSON.stringify(oldValue || []) === JSON.stringify(nextValue || [])) return;
        foundChanges = true;
        const change = el('p', 'proposal-change'); change.append(el('strong', '', `${label}: `), document.createTextNode(`${pretty(oldValue)} → ${pretty(nextValue)}`)); box.append(change);
      };
      compare('Цель', oldGoal, nextGoal);
      compare('Темы профиля', before.rag_context?.current_topics || [], profile.rag_context?.current_topics || []);
      compare('Интересы', before.interests || [], profile.interests || []);
      compare('Класс', before.grade || '', profile.grade || '');
    });
    const memoryCount = appendProposalMemory(box, data);
    if (!foundChanges && !memoryCount) box.append(el('p', '', 'В предложенном профиле нет отличий в цели, темах, интересах и классе. Проверьте заметку перед подтверждением.'));
    box.append(el('p', '', 'После подтверждения показанные изменения попадут в учебный профиль.'));
    const confirm = el('button', 'button primary wide', 'Подтвердить изменение профиля'); confirm.type = 'button'; confirm.addEventListener('click', () => busy(confirm, async () => { await request(`/lessons/proposals/${encodeURIComponent(data.proposal_id || state.proposal)}/confirm`, { method: 'POST' }); toast('Изменение подтверждено'); box.hidden = true; await selectLearner(state.learnerId); })); box.append(confirm);
  }
  function chatBubble(role, text, tool) {
    const bubble = el('div', `chat-bubble ${role}`, text);
    if (role === 'assistant' && tool?.operation === 'print_output' && tool.status === 'verified' && typeof tool.facts?.stdout === 'string') {
      const stdout = tool.facts.stdout;
      bubble.replaceChildren(el('p', '', 'Проверенный вывод:'), el('pre', 'learning-tool-stdout', stdout));
      bubble.append(el('p', 'field-hint', !stdout ? 'Вывод пустой.' : stdout.endsWith('\n') ? 'В конце есть перевод строки.' : 'В конце нет перевода строки.'));
    }
    return bubble;
  }
  function updateChat(role, text, citations = [], tool = null) {
    const bubble = chatBubble(role, text, tool);
    if (role === 'assistant' && citations.length) {
      const tags = el('div', 'chat-citations'); tags.append(document.createTextNode('Материалы: '));
      citations.slice(0, 5).forEach(item => tags.append(el('span', '', citationLabel(item))));
      bubble.append(tags);
    }
    $('#chat-log').append(bubble);
    if (state.learnerId) { const history = state.chatByLearner.get(state.learnerId) || []; history.push({ role, content: text, tool }); state.chatByLearner.set(state.learnerId, history.slice(-40)); }
    bubble.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  function citationLabel(item) {
    const context = `${item?.subject_id || ''} ${item?.topic_id || ''} ${item?.source_file || ''}`.toLocaleLowerCase('ru');
    if (/python|_py|py_/.test(context)) return 'Материал по Python';
    if (/scratch/.test(context)) return 'Материал по Scratch';
    if (/math|математ/.test(context)) return 'Учебный материал по математике';
    if (/programming|программ/.test(context)) return 'Учебный материал по программированию';
    return 'Учебный материал';
  }
  function renderChatHistory(learnerId) {
    const root = $('#chat-log'); root.replaceChildren(el('div', 'chat-bubble assistant', greeting));
    (state.chatByLearner.get(learnerId) || []).forEach(message => {
      const bubble = chatBubble(message.role, message.content, message.tool);
      const status = message.role === 'user' && ({ pending:'Ответ ещё готовится.', unknown_outcome:'Результат запроса неизвестен.', failed:'Сообщение не получило ответа.' })[message.delivery_status];
      if (status) bubble.append(el('small', 'field-hint', status));
      root.append(bubble);
    });
  }
  function recoveryBlocksMessages() {
    if (state.account?.role !== 'student') return false;
    const recovery = sessionRecovery.get(state.learnerId);
    return !state.learnerId || sessionClosures.has(state.learnerId) || !recovery || recovery.loading || !!recovery.error ||
      state.learningSessionsByLearner.get(state.learnerId)?.status === 'active' && !!recovery.unresolved_status;
  }
  function validCurrentSession(data) {
    if (data?.available === false) return true;
    return data?.available === true && typeof data.learning_session_id === 'string' && !!data.learning_session_id &&
      typeof data.topic_id === 'string' && !!data.topic_id && typeof data.topic_title === 'string' &&
      ['math', 'python', 'scratch', 'programming', 'general'].includes(data.subject_id) && data.status === 'active' &&
      typeof data.history_truncated === 'boolean' && [null, 'pending', 'unknown_outcome'].includes(data.unresolved_status) &&
      Array.isArray(data.messages) && data.messages.length <= 40 && data.messages.every(message =>
        ['user', 'assistant'].includes(message?.role) && typeof message.content === 'string' &&
        (message.delivery_status === undefined || ['completed', 'failed', 'pending', 'unknown_outcome'].includes(message.delivery_status)));
  }
  async function restoreLearningSession() {
    if (state.account?.role !== 'student' || !state.learnerId || sessionClosures.has(state.learnerId) || state.pendingMessages.has(state.learnerId)) return;
    const learnerId = state.learnerId, token = state.token, viewRevision = state.viewRevision;
    const recovery = { loading:true };
    sessionRecovery.set(learnerId, recovery); guidedViewRevision++; chatReturnAttempt = null;
    renderLearningSession();
    const currentView = () => token === state.token && learnerId === state.learnerId && viewRevision === state.viewRevision &&
      state.account?.role === 'student' && sessionRecovery.get(learnerId) === recovery && !state.pendingMessages.has(learnerId);
    try {
      const data = await request(`/learners/${encodeURIComponent(learnerId)}/learning-sessions/current`);
      if (!currentView()) return;
      if (!validCurrentSession(data)) throw new Error('Сервер не смог восстановить занятие. Обнови его состояние.');
      if (data.available) {
        state.learningSessionsByLearner.set(learnerId, { learning_session_id:data.learning_session_id,
          topic_id:data.topic_id, topic_title:data.topic_title, subject_id:data.subject_id, status:'active' });
        state.chatByLearner.set(learnerId, data.messages.map(message => ({ role:message.role, content:message.content, delivery_status:message.delivery_status })));
        sessionRecovery.set(learnerId, { loading:false, unresolved_status:data.unresolved_status, history_truncated:data.history_truncated });
      } else {
        const previous = state.learningSessionsByLearner.get(learnerId);
        if (previous?.status === 'active') state.learningSessionsByLearner.set(learnerId, { ...previous, status:'completed', ended_elsewhere:true });
        sessionRecovery.set(learnerId, { loading:false, unresolved_status:null, history_truncated:false });
      }
      renderChatHistory(learnerId); renderLearningSession();
      if (state.dashboard) refreshGuided();
    } catch (error) {
      if (!currentView()) return;
      sessionRecovery.set(learnerId, { loading:false, error:errorText(error) });
      renderLearningSession();
    }
  }
  function resetLearningArchive() {
    learningArchive = null; $('#learning-archive').open = false;
    $('#learning-archive-list').replaceChildren(); $('#learning-archive-log').replaceChildren();
    $('#learning-archive-status').textContent = ''; $('#learning-archive-detail-status').textContent = '';
    $('#learning-archive-view').hidden = true; $('#more-learning-sessions').hidden = true;
    $('#older-learning-messages').hidden = true; $('#refresh-learning-archive').disabled = false;
  }
  function archiveIsCurrent(view) {
    return view === learningArchive && view.token === state.token && view.learnerId === state.learnerId &&
      view.viewRevision === state.viewRevision && state.account?.role === 'student';
  }
  function archiveDate(value) {
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? new Intl.DateTimeFormat('ru-RU', { day:'numeric', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit' }).format(date) : 'Дата не указана';
  }
  function validArchiveCursor(cursor) { return cursor === null || typeof cursor === 'string' && cursor.length > 0 && cursor.length <= 1024; }
  function validSavedMessages(messages) {
    return Array.isArray(messages) && messages.length <= 40 && messages.every(message =>
      ['user', 'assistant'].includes(message?.role) && typeof message.content === 'string' &&
      (message.delivery_status === undefined || ['completed', 'failed', 'pending', 'unknown_outcome'].includes(message.delivery_status)));
  }
  function renderLearningArchive() {
    const view = learningArchive;
    if (!view || !archiveIsCurrent(view)) return;
    $('#refresh-learning-archive').disabled = view.listLoading;
    $('#learning-archive-status').textContent = view.listError || (view.listLoading ? 'Загружаем прошлые занятия…' : view.loaded && !view.sessions.length ? 'Завершённых занятий пока нет.' : '');
    const list = $('#learning-archive-list'); list.replaceChildren();
    view.sessions.forEach(session => {
      const button = el('button', 'learning-archive-item'); button.type = 'button';
      button.setAttribute('aria-pressed', String(view.selection?.id === session.learning_session_id));
      button.append(el('span', '', session.topic_title), el('small', '', archiveDate(session.completed_at)));
      button.addEventListener('click', () => openArchivedSession(session)); list.append(button);
    });
    $('#more-learning-sessions').hidden = !view.next_cursor; $('#more-learning-sessions').disabled = view.listLoading;
    const selected = view.selection; $('#learning-archive-view').hidden = !selected;
    if (!selected) { $('#learning-archive-log').replaceChildren(); return; }
    $('#learning-archive-title').textContent = selected.title;
    $('#learning-archive-detail-status').textContent = selected.error || (selected.loading ? 'Загружаем переписку…' : selected.olderLoading ? 'Загружаем ранние сообщения…' : selected.messages.length ? 'Завершённое занятие. Это просмотр прошлой переписки.' : 'В этом занятии нет доступных сообщений.');
    $('#older-learning-messages').hidden = !selected.next_cursor; $('#older-learning-messages').disabled = selected.loading || selected.olderLoading;
    const log = $('#learning-archive-log'); log.replaceChildren();
    selected.messages.forEach(message => {
      const bubble = chatBubble(message.role, message.content);
      const status = message.role === 'user' && ({ pending:'Ответ не был завершён.', unknown_outcome:'Результат запроса остался неизвестным.', failed:'Сообщение не получило ответа.' })[message.delivery_status];
      if (status) bubble.append(el('small', 'field-hint', status));
      log.append(bubble);
    });
  }
  async function loadLearningArchive(more = false) {
    if (state.account?.role !== 'student' || !state.learnerId) return;
    if (!learningArchive) learningArchive = { learnerId:state.learnerId, token:state.token, viewRevision:state.viewRevision,
      sessions:[], loaded:false, next_cursor:null, listLoading:false, selection:null };
    const view = learningArchive;
    if (!archiveIsCurrent(view) || view.listLoading || more && !view.next_cursor) return;
    const before = more ? view.next_cursor : null;
    view.listLoading = true; view.listError = null; renderLearningArchive();
    try {
      const query = before ? `?${new URLSearchParams({ before })}` : '';
      const data = await request(`/learners/${encodeURIComponent(view.learnerId)}/learning-sessions${query}`);
      if (!archiveIsCurrent(view)) return;
      const valid = Array.isArray(data.sessions) && data.sessions.length <= 10 && validArchiveCursor(data.next_cursor) &&
        data.sessions.every(session => typeof session?.learning_session_id === 'string' && !!session.learning_session_id &&
          typeof session.topic_id === 'string' && typeof session.topic_title === 'string' && typeof session.subject_id === 'string' && typeof session.completed_at === 'string');
      if (!valid || before && data.next_cursor === before) throw new Error('Список занятий не удалось прочитать. Обнови список.');
      const sessions = more ? [...view.sessions, ...data.sessions] : data.sessions;
      if (new Set(sessions.map(session => session.learning_session_id)).size !== sessions.length) throw new Error('Список занятий изменился. Обнови список.');
      view.sessions = sessions; view.next_cursor = data.next_cursor; view.loaded = true;
    } catch (error) { if (archiveIsCurrent(view)) view.listError = 'Не удалось загрузить прошлые занятия. Нажми «Обновить список», чтобы попробовать ещё раз.'; }
    finally { if (archiveIsCurrent(view)) { view.listLoading = false; renderLearningArchive(); } }
  }
  function openArchivedSession(session) {
    const view = learningArchive;
    if (!view || !archiveIsCurrent(view)) return;
    view.selection = { id:session.learning_session_id, title:session.topic_title, messages:[], next_cursor:null, loading:false };
    loadArchivedMessages();
  }
  async function loadArchivedMessages(older = false) {
    const view = learningArchive, selected = view?.selection;
    if (!view || !archiveIsCurrent(view) || !selected || selected.loading || selected.olderLoading || older && !selected.next_cursor) return;
    const before = older ? selected.next_cursor : null;
    selected[older ? 'olderLoading' : 'loading'] = true; selected.error = null; renderLearningArchive();
    const current = () => archiveIsCurrent(view) && view.selection === selected;
    try {
      const query = before ? `?${new URLSearchParams({ before })}` : '';
      const data = await request(`/learners/${encodeURIComponent(view.learnerId)}/learning-sessions/${encodeURIComponent(selected.id)}/history${query}`);
      if (!current()) return;
      if (data.learning_session_id !== selected.id || data.status !== 'completed' || typeof data.topic_title !== 'string' ||
          !validSavedMessages(data.messages) || !validArchiveCursor(data.next_cursor) || before && data.next_cursor === before) throw new Error('Переписку не удалось прочитать. Открой занятие ещё раз.');
      selected.title = data.topic_title; selected.messages = older ? [...data.messages, ...selected.messages] : data.messages;
      selected.next_cursor = data.next_cursor;
    } catch (error) {
      if (current()) {
        selected.error = [403, 404, 409].includes(error.status) ? 'Эта переписка сейчас недоступна. Обнови список занятий.' : 'Не удалось загрузить переписку. Открой занятие ещё раз.';
        if ([403, 404, 409].includes(error.status)) { selected.messages = []; selected.next_cursor = null; }
      }
    } finally {
      if (current()) { selected.loading = false; selected.olderLoading = false; renderLearningArchive(); }
    }
  }
  const actionLabels = { division_check: 'Проверить деление', division_places: 'Разряды частного', division_practice: 'Дать пример', print_output: 'Вывод Python', division_guided: 'Решать по шагам' };
  function guidedKey() { return `${state.learnerId}:${$('#chat-topic').value}`; }
  function canonicalGuidedKey() { return `${state.learnerId}:${guidedTopic}`; }
  function canReturnFromChat() {
    if (state.account?.role !== 'student' || !state.learnerId || recoveryBlocksMessages() || $('#chat-mode').value !== 'chat' || $('#chat-subject').value !== 'math' || state.pendingMessages.has(state.learnerId)) return false;
    const session = state.learningSessionsByLearner.get(state.learnerId);
    if (session?.status === 'active' && session.topic_id !== guidedTopic) return false;
    const chosen = state.chatTopicsByLearnerSubject.get(`${state.learnerId}:math`);
    return !chosen || chosen === guidedTopic;
  }
  function usableGuidedBookmark(entry) {
    const cursor = entry?.guided_division;
    return entry?.available === true && !entry.loading && !entry.error &&
      ['current_session', 'approved_memory'].includes(entry.origin) &&
      typeof entry.source_learning_session_id === 'string' && entry.source_learning_session_id.length > 0 &&
      typeof cursor?.state_id === 'string' && cursor.state_id.length > 0 &&
      ['active', 'paused'].includes(cursor.status) &&
      Number.isSafeInteger(cursor.dividend) && cursor.dividend > 0 && cursor.dividend <= 999999 &&
      Number.isSafeInteger(cursor.divisor) && cursor.divisor > 0 && cursor.divisor <= 99;
  }
  function bookmarkIdentity(entry) {
    const cursor = entry?.guided_division || {};
    return JSON.stringify([entry?.origin, entry?.source_learning_session_id, entry?.active_learning_session_id || null,
      ...['version', 'state_id', 'dividend', 'divisor', 'step_index', 'phase', 'status'].map(key => cursor[key])]);
  }
  function renderChatReturn(pending) {
    const entry = guidedContexts.get(canonicalGuidedKey());
    const shown = canReturnFromChat() && usableGuidedBookmark(entry);
    $('#chat-guided-return').hidden = !shown;
    $('#chat-guided-return-label').textContent = shown ? `${topicTitle(topicById(guidedTopic) || { title: guidedTopicTitle })} · ${entry.guided_division.dividend} : ${entry.guided_division.divisor}` : '';
    $('#chat-guided-return-message').textContent = !shown ? '' : entry.origin === 'approved_memory'
      ? 'Записи прошлого занятия просмотрены учителем. Можно вернуться к сохранённому примеру.'
      : 'Последний проверенный шаг сохранён. Можно вернуться к нему, когда захочешь.';
    $('#chat-guided-return-button').disabled = !shown || !!pending || !!chatReturnAttempt;
  }
  function renderGuided(pending) {
    const shown = $('#chat-mode').value === 'division_guided';
    $('#guided-division-panel').hidden = !shown;
    if (!shown) return;
    const entry = guidedContexts.get(guidedKey()), cursor = entry?.guided_division;
    const ready = entry?.available && cursor && !entry.loading;
    const answering = ready && !entry.resume_required && entry.origin === 'current_session' && cursor.status === 'active';
    const resuming = ready && (entry.resume_required || entry.origin === 'approved_memory' || cursor.status === 'paused');
    const unavailableMessage = {
      teacher_review_required: 'Для продолжения учителю нужно проверить записи прошлого занятия. Пока можно начать новый пример.',
      corrected_guided_source: 'Запись сохранённого шага исправлена. Продолжение этого примера недоступно. Можно начать новый.',
      later_message_barrier: 'После этого шага выбрано другое учебное действие. Можно начать новый пример.',
      message_outcome_unavailable: 'Результат предыдущего сообщения пока не подтверждён. Продолжение недоступно.',
      invalid_guided_chain: 'Сохранённый шаг не удалось проверить. Можно начать новый пример.',
      another_actor: 'Сохранённого шага для этого входа нет. Можно начать новый пример.',
      exercise_completed: 'Этот пример завершён. Можно начать новый.'
    }[entry?.reason];
    $('#guided-division-status').textContent = entry?.loading ? 'Проверяем сохранённый шаг…' : entry?.error ||
      (ready ? entry.resume_required ? 'Последний проверенный шаг сохранён. После беседы можно вернуться к нему.' : entry.origin === 'approved_memory' ? 'Есть просмотренный учителем пример. Продолжи его явно или начни новый.' : cursor.status === 'paused' ? 'Пример на паузе.' : 'Один вопрос – один шаг. Готовый ответ не показываем.' : unavailableMessage || (cursor?.status === 'completed' ? 'Этот пример завершён. Можно начать новый.' : 'Сохранённого шага для продолжения нет. Можно начать новый пример.'));
    $('#guided-division-example').hidden = !cursor;
    $('#guided-division-example').textContent = cursor ? `${cursor.dividend} : ${cursor.divisor}` : '';
    $('#guided-division-prompt').hidden = !entry?.prompt;
    $('#guided-division-prompt').textContent = entry?.prompt || '';
    $('#guided-division-answer-form').hidden = !answering;
    $('#guided-division-hint').hidden = !answering;
    $('#guided-division-pause').hidden = !answering;
    $('#guided-division-resume').hidden = !resuming;
    $('#guided-division-resume').textContent = entry?.resume_required ? 'Вернуться к сохранённому шагу' : 'Продолжить пример';
    $('#guided-division-refresh').hidden = false;
    $$('#guided-division-panel input, #guided-division-panel button').forEach(field => { field.disabled = !!pending || !!entry?.loading || recoveryBlocksMessages(); });
  }
  async function refreshGuided() {
    const mode = $('#chat-mode').value, inGuided = mode === 'division_guided';
    if (state.account?.role !== 'student' || !state.learnerId || recoveryBlocksMessages() || !inGuided && !canReturnFromChat()) return null;
    const learnerId = state.learnerId, key = canonicalGuidedKey(), token = state.token, revision = ++guidedViewRevision;
    if (inGuided && $('#chat-topic').value !== guidedTopic) { renderLearningSession(); return null; }
    const query = new URLSearchParams({ topic_id: guidedTopic });
    const session = state.learningSessionsByLearner.get(learnerId);
    const sessionIdentity = () => {
      const current = state.learningSessionsByLearner.get(learnerId);
      return JSON.stringify([current?.learning_session_id, current?.status, current?.topic_id]);
    };
    const priorSession = sessionIdentity(), viewRevision = state.viewRevision, subject = $('#chat-subject').value, topic = $('#chat-topic').value;
    const currentView = () => revision === guidedViewRevision && token === state.token && learnerId === state.learnerId &&
      viewRevision === state.viewRevision && mode === $('#chat-mode').value && subject === $('#chat-subject').value &&
      topic === $('#chat-topic').value && priorSession === sessionIdentity() && !state.pendingMessages.has(learnerId);
    if (session?.status === 'active') query.set('learning_session_id', session.learning_session_id);
    guidedContexts.set(key, { loading: true }); renderLearningSession();
    try {
      const data = await request(`/learners/${encodeURIComponent(learnerId)}/guided-division?${query}`);
      if (!currentView()) return null;
      guidedContexts.set(key, data);
      if (inGuided && data.active_learning_session_id) state.learningSessionsByLearner.set(learnerId, { learning_session_id: data.active_learning_session_id, topic_id: guidedTopic, topic_title: guidedTopicTitle, subject_id: 'math', status: 'active' });
      renderLearningSession();
      return data;
    } catch (error) {
      if (!currentView()) return null;
      guidedContexts.set(key, { error: errorText(error) });
      renderLearningSession();
      return null;
    }
  }
  async function returnFromChat() {
    const entry = guidedContexts.get(canonicalGuidedKey());
    if (chatReturnAttempt || !canReturnFromChat() || !usableGuidedBookmark(entry)) return;
    const attempt = {}, prior = bookmarkIdentity(entry);
    chatReturnAttempt = attempt; renderLearningSession();
    try {
      const data = await refreshGuided();
      if (chatReturnAttempt !== attempt || !canReturnFromChat()) return;
      if (!usableGuidedBookmark(data)) { toast('Сохранённый шаг сейчас недоступен. Можно продолжить разговор.'); return; }
      if (bookmarkIdentity(data) !== prior) { toast('Сохранённый шаг изменился. Проверь пример и нажми ещё раз.'); return; }
      state.chatTopicsByLearnerSubject.set(`${state.learnerId}:math`, guidedTopic);
      if (data.active_learning_session_id) state.learningSessionsByLearner.set(state.learnerId, {
        learning_session_id: data.active_learning_session_id, topic_id: guidedTopic, topic_title: guidedTopicTitle, subject_id: 'math', status: 'active'
      });
      guidedViewRevision++;
      $('#chat-mode').value = 'division_guided'; renderLearningSession();
      sendGuided('division_guided_resume');
    } finally {
      if (chatReturnAttempt === attempt) { chatReturnAttempt = null; renderLearningSession(); }
    }
  }
  function sendGuided(operation, value) {
    const learnerId = state.learnerId, entry = guidedContexts.get(guidedKey());
    if (!entry?.available || recoveryBlocksMessages() || state.pendingMessages.has(learnerId) || entry.resume_required && operation !== 'division_guided_resume') return;
    const action = { operation, expected_state_id: entry.guided_division.state_id };
    if (operation === 'division_guided_answer') action.value = value;
    if (operation === 'division_guided_resume' && entry.origin === 'approved_memory') action.source_learning_session_id = entry.source_learning_session_id;
    const content = operation === 'division_guided_answer' ? `Мой ответ на шаг: ${value}.` : ({division_guided_hint:'Хочу подсказку к этому шагу.', division_guided_pause:'Сделаем паузу.', division_guided_resume:'Продолжим сохранённый пример.'})[operation];
    if (state.learningSessionsByLearner.get(learnerId)?.status === 'completed') { state.chatByLearner.delete(learnerId); renderChatHistory(learnerId); }
    state.pendingMessages.set(learnerId, { content, learning_action: Object.freeze(action), client_message_id: crypto.randomUUID(), client_session_id: crypto.randomUUID(), subject_id: 'math', topic_id: guidedTopic });
    updateChat('user', content); sendPendingMessage();
  }
  function resetLearningAction() {
    $('#learning-action-form').reset(); $('#chat-mode').value = 'chat';
    $('#learning-action-error').hidden = true; $('#learning-action-error').textContent = '';
    $('#action-code-count').textContent = '0 из 4000 символов';
  }
  function chatTopicSubject(topic) {
    const id = topic?.topic_id || topic?.id || '';
    if (id.startsWith('programming.python.')) return 'python';
    if (id.startsWith('programming.scratch.')) return 'scratch';
    return topic?.subject_id || id.split('.')[0];
  }
  function renderChatTopicOptions() {
    const select = $('#chat-topic'), session = state.learningSessionsByLearner.get(state.learnerId), pending = state.pendingMessages.get(state.learnerId);
    const active = session?.status === 'active';
    $$('#chat-subject option[data-restored-subject]').forEach(option => { if (!active || option.value !== session.subject_id) option.remove(); });
    if (active && session.subject_id && ![...$('#chat-subject').options].some(option => option.value === session.subject_id)) {
      const option = new Option(subjectTitles[session.subject_id] || 'Учебное занятие', session.subject_id);
      option.dataset.restoredSubject = 'true'; $('#chat-subject').add(option);
    }
    if (active && session.subject_id) $('#chat-subject').value = session.subject_id;
    if (pending?.subject_id) $('#chat-subject').value = pending.subject_id;
    const subject = $('#chat-subject').value, key = `${state.learnerId}:${subject}`;
    const matches = topic => chatTopicSubject(topic) === subject || subject === 'programming' && ['python', 'scratch'].includes(chatTopicSubject(topic));
    const goal = state.dashboard?.profile?.goal;
    const candidates = state.topics.filter(matches);
    if (goal?.topic_id && goal.subject_id && matches(goal) && !candidates.some(topic => (topic.topic_id || topic.id) === goal.topic_id)) candidates.unshift({ ...goal, title: goal.topic });
    select.replaceChildren(new Option('Общая беседа по предмету', subject));
    candidates.forEach(topic => select.add(new Option(`${topicTitle(topic)}${topic.grade ? ` · ${topic.grade} класс` : ''}`, topic.topic_id || topic.id)));
    const chosen = active ? session.topic_id : pending?.topic_id || state.chatTopicsByLearnerSubject.get(key) || (goal?.topic_id && matches(goal) ? goal.topic_id : subject);
    if (chosen && ![...select.options].some(option => option.value === chosen) && (active || chosen === guidedTopic && $('#chat-mode').value === 'division_guided')) {
      select.add(new Option(chosen === guidedTopic ? guidedTopicTitle : topicTitle(session), chosen));
    }
    select.value = [...select.options].some(option => option.value === chosen) ? chosen : subject;
    select.disabled = active || !!pending || recoveryBlocksMessages();
  }
  function renderLearningAction(pending) {
    const subject = $('#chat-subject').value;
    $$('#chat-mode option').forEach(option => {
      option.disabled = option.value === 'print_output' ? !['python', 'programming'].includes(subject) : option.value !== 'chat' && subject !== 'math';
      if (option.value === 'division_guided' && state.learningSessionsByLearner.get(state.learnerId)?.status === 'active' && $('#chat-topic').value !== guidedTopic) option.disabled = true;
    });
    if ($('#chat-mode').selectedOptions[0]?.disabled && !pending) $('#chat-mode').value = 'chat';
    const mode = $('#chat-mode').value, isAction = mode !== 'chat';
    $('#chat-form').hidden = isAction; $('#learning-action-panel').hidden = !isAction;
    $('#learning-action-availability').hidden = subject !== 'scratch';
    $('#action-division-fields').hidden = !['division_check', 'division_places', 'division_guided'].includes(mode);
    $('#action-quotient-field').hidden = mode !== 'division_check';
    $('#action-practice-fields').hidden = mode !== 'division_practice';
    $('#action-print-fields').hidden = mode !== 'print_output';
    $('#learning-action-title').textContent = actionLabels[mode] || '';
    $('#send-learning-action').textContent = pending?.sending ? 'Проверяем…' : mode === 'division_guided' ? 'Начать новый пример' : actionLabels[mode] || 'Выполнить';
    $('#learning-action-hint').textContent = {
      division_check: 'Покажу результат и проверку умножением. Можно указать свой ответ. Для деления с остатком покажу остаток.',
      division_places: 'Покажу всё частное и значение каждой его цифры.',
      division_practice: 'Деление без остатка. В частном три цифры, ноль в десятках и ненулевые единицы. Ответ останется тебе.',
      division_guided: 'По одному шагу, без готового ответа. Поддерживается положительное деление без остатка. Проверка шага не означает освоение темы.',
      print_output: 'Покажу точные символы вывода. Поддерживаются до 8 вызовов print со строками, целыми числами, True, False, None и параметрами sep и end. Выражения и переменные здесь не поддерживаются.'
    }[mode] || '';
    $('#chat-mode').disabled = !!pending || recoveryBlocksMessages();
    $$('#learning-action-form input, #learning-action-form select, #learning-action-form textarea, #learning-action-form button').forEach(field => { field.disabled = !!pending || recoveryBlocksMessages(); });
    renderGuided(pending);
    renderChatReturn(pending);
  }
  function renderLearningSession() {
    renderChatTopicOptions();
    const session = state.learningSessionsByLearner.get(state.learnerId), pending = state.pendingMessages.get(state.learnerId);
    const recovery = sessionRecovery.get(state.learnerId), blocked = recoveryBlocksMessages();
    const title = session ? topicTitle({ ...session, topic_id: session.topic_id || session.subject_id }) : '';
    $('#learning-session-status').textContent = session?.status === 'active' ? `${title} · Занятие идёт. После завершения диалог будет сохранён для просмотра учителем.` : session?.ended_elsewhere ? `${title} · Занятие больше не активно. Следующее сообщение начнёт новое.` : session?.status === 'completed' ? `${title} · Занятие завершено. Обновление памяти поставлено в очередь.` : `${subjectTitles[$('#chat-subject').value] || 'Учебное занятие'} · Занятие начнётся с первого вопроса.`;
    const notes = [];
    if (sessionClosures.has(state.learnerId)) notes.push('Сохраняем завершение занятия…');
    else if (recovery?.loading) notes.push('Восстанавливаем занятие и переписку…');
    else if (recovery?.error) notes.push('Не удалось проверить текущее занятие. Нажми «Обновить занятие», чтобы проверить ещё раз.');
    else if (session?.status === 'active' && recovery?.unresolved_status === 'pending') notes.push('Предыдущий ответ ещё готовится. Обнови занятие, чтобы проверить результат. Новый вопрос пока недоступен.');
    else if (session?.status === 'active' && recovery?.unresolved_status === 'unknown_outcome') notes.push('Результат предыдущего запроса неизвестен. Можно обновить занятие или завершить его. Запрос повторно не отправляется.');
    if (recovery?.history_truncated) notes.push('Показана последняя часть переписки. Ранние сообщения сохранены.');
    $('#learning-session-recovery-note').textContent = notes.join(' '); $('#learning-session-recovery-note').hidden = !notes.length;
    $('#refresh-learning-session').disabled = !state.learnerId || !!recovery?.loading || !!pending || sessionClosures.has(state.learnerId);
    $('#complete-learning-session').hidden = session?.status !== 'active';
    $('#complete-learning-session').disabled = !!pending && !pending.terminal || !!recovery?.loading || !!recovery?.error || recovery?.unresolved_status === 'pending' || sessionClosures.has(state.learnerId);
    $('#chat-subject').disabled = session?.status === 'active' || !!pending || blocked;
    $('#send-chat').disabled = !!pending || blocked; $('#chat-input').disabled = !!pending || blocked;
    $('#chat-retry').hidden = !pending?.error; $('#chat-error').textContent = pending?.error || ''; $('#retry-chat').hidden = !!pending?.terminal;
    renderLearningAction(pending);
  }
  async function sendPendingMessage() {
    const learnerId = state.learnerId, pending = state.pendingMessages.get(learnerId), token = state.token;
    if (!pending || pending.sending) return;
    let ordinaryReplyReady = false;
    pending.sending = true; pending.error = null; renderLearningSession(); $('#send-chat').textContent = 'Ищем объяснение…';
    try {
      let session = state.learningSessionsByLearner.get(learnerId);
      if (!session || session.status !== 'active') {
        session = await request(`/learners/${encodeURIComponent(learnerId)}/learning-sessions`, { method: 'POST', body: { topic_id: pending.topic_id || pending.subject_id, client_session_id: pending.client_session_id } });
        if (state.token !== token) return;
        session.subject_id = pending.subject_id; state.learningSessionsByLearner.set(learnerId, session);
        sessionRecovery.set(learnerId, { loading:false, unresolved_status:null, history_truncated:false });
      }
      const body = { model: 'teachcopilot-rag', learning_session_id: session.learning_session_id, client_message_id: pending.client_message_id, messages: pending.learning_action ? [] : [{ role: 'user', content: pending.content }], metadata: { subject_id: session.subject_id || pending.subject_id } };
      if (pending.learning_action) body.learning_action = pending.learning_action;
      const data = await request('/v1/chat/completions', { method: 'POST', body });
      if (state.token !== token) return;
      const message = data.choices?.[0]?.message?.content;
      if (!message) throw new Error('В ответе нет объяснения. Повторите этот же вопрос, чтобы проверить результат запроса.');
      guidedViewRevision++; chatReturnAttempt = null;
      const guideKey = `${learnerId}:${session.topic_id}`;
      if (data.guided_division) guidedContexts.set(guideKey, { available: data.guided_division.status !== 'completed', origin: 'current_session', guided_division: data.guided_division, source_learning_session_id: session.learning_session_id, prompt: message });
      else { guidedContexts.delete(guideKey); ordinaryReplyReady = true; }
      state.pendingMessages.delete(learnerId);
      if (state.learnerId === learnerId) updateChat('assistant', message, data.citations || data.sources || [], data.learning_tool);
      else { const history = state.chatByLearner.get(learnerId) || []; history.push({ role: 'assistant', content: message, tool: data.learning_tool }); state.chatByLearner.set(learnerId, history); }
    } catch (error) {
      if (state.token === token) {
        if (pending.learning_action && ([400, 422].includes(error.status) || error.status === 409 && pending.learning_action.operation.startsWith('division_guided_') && !/automatic retry is disabled|already processing/.test(error.detail || ''))) {
          state.pendingMessages.delete(learnerId);
          const history = state.chatByLearner.get(learnerId) || []; history.pop(); state.chatByLearner.set(learnerId, history);
          if (state.learnerId === learnerId) {
            renderChatHistory(learnerId); $('#learning-action-error').textContent = error.detail || errorText(error); $('#learning-action-error').hidden = false;
            if (error.status === 409) { $('#learning-action-error').textContent = 'Сохранённый шаг изменился. Обновите его перед следующим ответом.'; await refreshGuided(); }
          }
        } else {
          pending.terminal = error.status === 409 && /automatic retry is disabled/.test(error.detail || '');
          pending.error = pending.terminal ? 'Исход предыдущего запроса неизвестен. Завершите занятие; новый оплачиваемый запрос автоматически не отправляется.' : `${errorText(error)} Можно повторно проверить этот же вопрос.`;
        }
        toast(errorText(error));
      }
    }
    finally {
      pending.sending = false;
      if (state.token === token && state.learnerId === learnerId) { $('#send-chat').textContent = 'Отправить вопрос'; renderLearningSession(); if (ordinaryReplyReady) refreshGuided(); }
    }
  }

  async function loginWith(path, body, headers = {}) {
    if (loginBusy) return;
    loginBusy = true;
    const button = $('#login-form button[type="submit"]'), errorNode = $('#login-error');
    errorNode.hidden = true;
    $$('#login-form button, #login-form input').forEach(field => { field.disabled = true; });
    button.textContent = 'Входим…';
    try {
      const data = await request(path, { method: 'POST', body, headers });
      state.token = data.access_token; sessionStorage.setItem(tokenKey, state.token);
      await boot();
    } catch (error) {
      errorNode.textContent = path.startsWith('/auth/demo/')
        ? 'Демонстрационный вход недоступен. Можно войти по выданной учётной записи.' : errorText(error);
      errorNode.hidden = false;
    }
    finally {
      loginBusy = false;
      $$('#login-form button, #login-form input').forEach(field => { field.disabled = false; });
      button.textContent = 'Войти';
    }
  }
  $('#login-form').addEventListener('submit', event => {
    event.preventDefault();
    loginWith('/auth/login', { email: $('#email').value.trim(), password: $('#password').value });
  });
  ['teacher', 'student'].forEach(role => $(`#login-as-${role}`).addEventListener('click', () => {
    loginWith(`/auth/demo/${role}`, {}, { 'X-TeachCopilot-Demo': '1' });
  }));
  $('#logout').addEventListener('click', async () => { try { await request('/auth/logout', { method: 'POST' }); } catch { /* local session is still cleared */ } signOutLocal(); });
  $('#learner-filter').addEventListener('input', filterRoster);
  $('#retry-load').addEventListener('click', () => state.learnerId ? selectLearner(state.learnerId) : loadLearners());
  $('#zpd-help').addEventListener('click', () => { const box = $('#zpd-explainer'), open = box.hidden; box.hidden = !open; $('#zpd-help').setAttribute('aria-expanded', String(open)); });
  $$('.mobile-tabs button').forEach(button => button.addEventListener('click', () => { $$('.mobile-tabs button').forEach(b => b.classList.toggle('active', b === button)); $('.workspace-grid').dataset.mobilePanel = button.dataset.panel; }));
  $('#task-form').addEventListener('submit', event => { event.preventDefault(); const button = $('#create-task'); busy(button, async () => { if (!state.learnerId) throw new Error('Сначала выберите ученика.'); const topic = $('#topic-select').value || null; if (!topic) throw new Error('Выберите тему задания.'); button.textContent = 'Готовим…'; const mode = $('#task-mode').value, learnerId = state.learnerId, viewRevision = state.viewRevision; const current = state.nextStepsByLearner.get(learnerId); const body = { learner_id: learnerId, topic_id: topic, count: Number($('#task-count').value), mode }; if (current?.useForTask && current.result?.topic_id === topic) body.next_step_id = current.result.next_step_id; const data = await request('/task/design', { method: 'POST', body }); if (viewRevision !== state.viewRevision) return; if (current) current.useForTask = false; renderTask(data); toast('Черновик заданий готов к проверке'); }); });
  $('#request-next-step').addEventListener('click', () => {
    const button = $('#request-next-step'); if (button.disabled) return;
    const learnerId = state.learnerId, viewRevision = state.viewRevision, current = state.nextStepsByLearner.get(learnerId);
    if (current?.job_id && !current.finished) { button.disabled = true; pollNextStep(learnerId, current.job_id, viewRevision); return; }
    busy(button, async () => {
      const topic = $('#topic-select').value; if (!learnerId || !topic) throw new Error('Выберите ученика и тему занятия.');
      const method = $('#memory-method').value; button.textContent = 'Ставим в очередь…';
      const submitting = { submitting: true, statusText: 'Ставим рекомендацию в очередь…' }, token = state.token;
      state.nextStepsByLearner.set(learnerId, submitting);
      let job;
      try { job = await request(`/learners/${encodeURIComponent(learnerId)}/next-step`, { method: 'POST', body: { topic_id: topic, method } }); }
      catch (error) { if (state.token === token && state.nextStepsByLearner.get(learnerId) === submitting) { if (current) state.nextStepsByLearner.set(learnerId, current); else state.nextStepsByLearner.delete(learnerId); } throw error; }
      if (state.token !== token) return;
      state.nextStepsByLearner.set(learnerId, { job_id: job.job_id, topic_id: topic, method, statusText: 'Рекомендация в очереди.' });
      if (viewRevision !== state.viewRevision) return;
      renderNextStep(); await pollNextStep(learnerId, job.job_id, viewRevision);
    });
  });
  $('#refresh-memory').addEventListener('click', loadMemory);
  $('#topic-select').addEventListener('change', () => { const current = state.nextStepsByLearner.get(state.learnerId); if (current) current.useForTask = false; loadMemory(); });
  $('#show-answers').addEventListener('change', () => { if (state.task) renderTask(state.task); });
  $('#observation-form').addEventListener('submit', event => { event.preventDefault(); const button = $('#save-observation'); busy(button, async () => { const topic_id = $('#observation-topic').value, task_id = $('#observation-task').value.trim(), assessment_pair_id = $('#assessment-pair').value.trim(); if (!topic_id) throw new Error('Выберите тему наблюдения.'); if (!task_id) throw new Error('Добавьте короткое название задания.'); button.textContent = 'Сохраняем…'; const body = { expected_revision: state.revision || 1, topic_id, task_id, difficulty: $('#observation-difficulty').value, solved: $('#solved').checked, support_level: $('#support-level').value, notes: $('#observation-notes').value.trim() || undefined }; if (assessment_pair_id) body.assessment_pair_id = assessment_pair_id; const data = await request(`/learners/${encodeURIComponent(state.learnerId)}/observations`, { method: 'POST', body }); state.revision = data.revision; $('#observation-notes').value = ''; $('#observation-task').value = ''; toast('Наблюдение сохранено'); await selectLearner(state.learnerId); }); });
  $('#lesson-form').addEventListener('submit', event => { event.preventDefault(); const button = $('#preview-lesson'); busy(button, async () => { const notes = $('#lesson-notes').value.trim(); if (!notes) throw new Error('Добавьте заметку о занятии.'); button.textContent = 'Готовим предложение…'; const data = await request('/learner/update', { method: 'POST', body: { notes, learner_ids: [state.learnerId], apply: false } }); addProposalSummary(data); $('#proposal-result').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }); });
  $('#edit-profile').addEventListener('click', () => $('#edit-dialog').showModal());
  $('#edit-form').addEventListener('submit', event => { if (event.submitter?.value === 'cancel') return; event.preventDefault(); const button = $('#save-profile'); busy(button, async () => { const p = state.dashboard.profile, rag = p.rag_context || {}; const card = structuredClone(p); card.rag_context = { ...rag, current_goal: { ...(rag.current_goal || {}), goal: $('#edit-goal').value.trim() } }; card.interests = $('#edit-interests').value.split(',').map(x => x.trim()).filter(Boolean); await request(`/learners/${encodeURIComponent(state.learnerId)}`, { method: 'PUT', body: { expected_revision: state.revision || 1, card } }); $('#edit-dialog').close(); toast('Профиль сохранён'); await selectLearner(state.learnerId); }); });
  $('#chat-form').addEventListener('submit', event => {
    event.preventDefault(); const learnerId = state.learnerId, input = $('#chat-input'), content = input.value.trim();
    if (!learnerId || !content || recoveryBlocksMessages() || state.pendingMessages.has(learnerId)) return;
    if (state.learningSessionsByLearner.get(learnerId)?.status === 'completed') { state.chatByLearner.delete(learnerId); renderChatHistory(learnerId); }
    const client_message_id = crypto.randomUUID();
    state.pendingMessages.set(learnerId, { content, client_message_id, client_session_id: crypto.randomUUID(), subject_id: $('#chat-subject').value, topic_id: $('#chat-topic').value });
    updateChat('user', content); input.value = ''; sendPendingMessage();
  });
  $('#chat-mode').addEventListener('change', () => {
    guidedViewRevision++; chatReturnAttempt = null; $('#learning-action-error').hidden = true;
    if ($('#chat-mode').value === 'division_guided' && state.learningSessionsByLearner.get(state.learnerId)?.status !== 'active') {
      state.chatTopicsByLearnerSubject.set(`${state.learnerId}:math`, guidedTopic);
    }
    renderLearningSession(); refreshGuided();
  });
  $('#chat-subject').addEventListener('change', () => { guidedViewRevision++; chatReturnAttempt = null; resetLearningAction(); renderLearningSession(); refreshGuided(); });
  $('#chat-topic').addEventListener('change', () => {
    state.chatTopicsByLearnerSubject.set(`${state.learnerId}:${$('#chat-subject').value}`, $('#chat-topic').value);
    guidedViewRevision++; chatReturnAttempt = null; renderLearningSession(); refreshGuided();
  });
  $('#action-code').addEventListener('input', () => { $('#action-code-count').textContent = `${$('#action-code').value.length} из 4000 символов`; });
  $('#learning-action-form').addEventListener('submit', event => {
    event.preventDefault(); const learnerId = state.learnerId, operation = $('#chat-mode').value;
    if (!learnerId || !actionLabels[operation] || recoveryBlocksMessages() || state.pendingMessages.has(learnerId)) return;
    $('#learning-action-error').hidden = true;
    try {
      const action = { operation }; let content;
      const number = (selector, label, min, max) => {
        const value = $(selector).value;
        if (!/^[0-9]+$/.test(value) || Number(value) < min || Number(value) > max) { $(selector).focus(); throw new Error(`${label}: введите целое число от ${min} до ${max}.`); }
        return Number(value);
      };
      if (['division_check', 'division_places', 'division_guided'].includes(operation)) {
        if (operation === 'division_guided') action.operation = 'division_guided_start';
        action.dividend = number('#action-dividend', 'Делимое', operation === 'division_guided' ? 1 : 0, 999999);
        action.divisor = number('#action-divisor', 'Делитель', 1, 99);
        if (operation === 'division_check' && $('#action-quotient').value !== '') action.stated_quotient = number('#action-quotient', 'Мой ответ', 0, 999999);
        content = `${actionLabels[operation]}: ${action.dividend} : ${action.divisor}${action.stated_quotient === undefined ? '' : ` = ${action.stated_quotient}`}.`;
      } else if (operation === 'division_practice') {
        action.divisor_digits = Number($('#action-divisor-digits').value);
        content = `Дать пример: ${$('#action-divisor-digits').selectedOptions[0].textContent}. В частном три цифры и ноль в десятках. Ответ не показывай.`;
      } else {
        action.code = $('#action-code').value;
        if (!action.code.trim() || action.code.length > 4000) throw new Error('Введите код: от 1 до 4000 символов.');
        content = `Покажи точный вывод Python:\n${action.code}`;
      }
      if (state.learningSessionsByLearner.get(learnerId)?.status === 'completed') { state.chatByLearner.delete(learnerId); renderChatHistory(learnerId); }
      state.pendingMessages.set(learnerId, { content, learning_action: Object.freeze(action), client_message_id: crypto.randomUUID(), client_session_id: crypto.randomUUID(), subject_id: $('#chat-subject').value, topic_id: $('#chat-topic').value });
      updateChat('user', content); sendPendingMessage();
    } catch (error) { $('#learning-action-error').textContent = errorText(error); $('#learning-action-error').hidden = false; }
  });
  $('#retry-chat').addEventListener('click', () => busy($('#retry-chat'), sendPendingMessage));
  $('#guided-division-answer-form').addEventListener('submit', event => {
    event.preventDefault(); const text = $('#guided-division-answer').value;
    if (!/^[0-9]{1,6}$/.test(text)) { $('#learning-action-error').textContent = 'Введите одно целое число от 0 до 999999.'; $('#learning-action-error').hidden = false; return; }
    $('#learning-action-error').hidden = true; $('#guided-division-answer').value = ''; sendGuided('division_guided_answer', Number(text));
  });
  $('#guided-division-hint').addEventListener('click', () => sendGuided('division_guided_hint'));
  $('#guided-division-pause').addEventListener('click', () => sendGuided('division_guided_pause'));
  $('#guided-division-resume').addEventListener('click', () => sendGuided('division_guided_resume'));
  $('#guided-division-refresh').addEventListener('click', refreshGuided);
  $('#chat-guided-return-button').addEventListener('click', returnFromChat);
  $('#refresh-learning-session').addEventListener('click', restoreLearningSession);
  $('#learning-archive').addEventListener('toggle', () => {
    if ($('#learning-archive').open && !learningArchive?.loaded) loadLearningArchive();
  });
  $('#refresh-learning-archive').addEventListener('click', () => loadLearningArchive());
  $('#more-learning-sessions').addEventListener('click', () => loadLearningArchive(true));
  $('#older-learning-messages').addEventListener('click', () => loadArchivedMessages(true));
  $('#close-learning-archive').addEventListener('click', () => {
    if (learningArchive && archiveIsCurrent(learningArchive)) { learningArchive.selection = null; renderLearningArchive(); }
  });
  $('#complete-learning-session').addEventListener('click', async () => {
    const learnerId = state.learnerId, session = state.learningSessionsByLearner.get(learnerId), token = state.token;
    const recovery = sessionRecovery.get(learnerId);
    if (!session || session.status !== 'active' || sessionClosures.has(learnerId) || state.pendingMessages.has(learnerId) && !state.pendingMessages.get(learnerId).terminal || !recovery || recovery.loading || recovery.error || recovery.unresolved_status === 'pending') return;
    const attempt = {}; sessionClosures.set(learnerId, attempt); guidedViewRevision++; chatReturnAttempt = null; renderLearningSession();
    try {
      const result = await request(`/learning-sessions/${encodeURIComponent(session.learning_session_id)}/complete`, { method: 'POST', body: {} });
      if (state.token !== token || sessionClosures.get(learnerId) !== attempt) return;
      if (state.learningSessionsByLearner.get(learnerId)?.learning_session_id !== session.learning_session_id) return;
      state.learningSessionsByLearner.set(learnerId, { ...session, status:result.status || 'completed', job_id:result.job_id });
      state.pendingMessages.delete(learnerId); sessionRecovery.set(learnerId, { ...recovery, unresolved_status:null });
      if (state.learnerId === learnerId) toast('Занятие сохранено. Память обновится в фоне.');
    } catch (error) { if (state.token === token && state.learnerId === learnerId) toast(errorText(error)); }
    finally {
      if (sessionClosures.get(learnerId) === attempt) {
        sessionClosures.delete(learnerId);
        if (state.token === token && state.learnerId === learnerId) { renderLearningSession(); refreshGuided(); }
      }
    }
  });
  boot();
})();
