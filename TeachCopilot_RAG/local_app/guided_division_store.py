"""Reconstruct a guided cursor from immutable typed actions, never chat prose."""
from copy import deepcopy

from .guided_division import start_state, validate_state, transition, render_prompt
from .explicit_learning_actions import GUIDED_OPERATIONS, validate_action

TOPIC = 'math.g4.numbers.division_by_1_2_digit'


def _saved_step_candidate(rows):
    """Offer a bookmark, never interpret ordinary dialogue as checked progress."""
    conversation_continued = False
    for row in reversed(rows):
        if row['status'] == 'failed':
            continue
        if row['status'] != 'completed':
            return None, conversation_continued, 'message_outcome_unavailable'
        response = row.get('response')
        if not isinstance(response, dict):
            return None, conversation_continued, 'invalid_guided_chain'
        if 'guided_division' in response:
            return row, conversation_continued, None
        action = response.get('learning_action')
        if ('_guided_division_provenance' in response or isinstance(action, dict)
                and str(action.get('operation', '')).startswith('division_guided_')):
            return None, conversation_continued, 'invalid_guided_chain'
        if 'learning_action' in response:
            return None, conversation_continued, 'later_message_barrier'
        conversation_continued = True
    return None, conversation_continued, None


def guided_context(store, learner_id, topic_id, actor_id, *, learning_session_id=None,
                   pending_client_message_id=None):
    unavailable = {'available': False, 'reason': 'no_saved_guided_step'}
    if topic_id != TOPIC:
        return dict(unavailable, reason='unsupported_topic')
    with store._connect() as connection, connection.cursor() as cur:
        def session(session_id):
            cur.execute('SELECT * FROM teachcopilot.learning_sessions WHERE id=%s AND learner_id=%s',
                        (session_id, learner_id))
            result = cur.fetchone()
            if result is None:
                raise KeyError(session_id)
            if str(result['actor_id']) != str(actor_id):
                raise PermissionError('learning session belongs to another actor')
            if result['topic_id'] != topic_id:
                raise ValueError('guided session topic mismatch')
            return dict(result)

        def messages(session_id):
            cur.execute('''SELECT m.*,s.content,s.author_role FROM teachcopilot.learning_messages m
                JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
                WHERE m.learning_session_id=%s AND m.learner_id=%s ORDER BY m.created_at,m.id''',
                        (session_id, learner_id))
            return [dict(row) for row in cur.fetchall()]

        origin = 'current_session'
        if not learning_session_id:
            cur.execute('''SELECT id FROM teachcopilot.learning_sessions WHERE learner_id=%s
                AND actor_id=%s AND topic_id=%s AND status='active' ORDER BY created_at DESC,id DESC LIMIT 1''',
                        (learner_id, str(actor_id), topic_id))
            active = cur.fetchone()
            if active:
                learning_session_id = str(active['id'])
        selected_session = session(learning_session_id) if learning_session_id else None
        if selected_session and selected_session['status'] == 'active':
            unavailable['active_learning_session_id'] = str(selected_session['id'])
        current_rows = messages(learning_session_id) if selected_session else []
        visible = [row for row in current_rows if not (row['status'] == 'pending'
                   and row['client_message_id'] == pending_client_message_id)]
        candidate, conversation_continued, blocked = (
            _saved_step_candidate(visible) if selected_session and selected_session['status'] == 'active'
            else (None, False, None))
        if blocked:
            return dict(unavailable, reason=blocked)
        if not selected_session or selected_session['status'] != 'active' or candidate is None:
            origin = 'approved_memory'
            cur.execute('''SELECT id,actor_id FROM teachcopilot.learning_sessions WHERE learner_id=%s
                AND topic_id=%s AND status='completed' ORDER BY completed_at DESC,id DESC LIMIT 1''',
                        (learner_id, topic_id))
            latest = cur.fetchone()
            if not latest:
                return unavailable
            if str(latest['actor_id']) != str(actor_id):
                return dict(unavailable, reason='another_actor')
            selected_session = session(str(latest['id']))
            candidate, previous_conversation, blocked = _saved_step_candidate(messages(str(selected_session['id'])))
            conversation_continued = conversation_continued or previous_conversation
            if blocked:
                return dict(unavailable, reason=blocked)
        if candidate is None:
            return unavailable
        source_ids, expected, seen = [], candidate, set()
        try:
            for _ in range(256):
                key = (str(expected['learning_session_id']), expected['client_message_id'])
                if key in seen:
                    raise ValueError('cyclic guided chain')
                seen.add(key)
                session(key[0])
                if expected['status'] != 'completed' or expected['author_role'] not in {'student', 'teacher'}:
                    raise ValueError('incomplete guided source')
                response = expected['response']
                state = validate_state(response.get('guided_division'))
                if state['state_id'] != expected['client_message_id']:
                    raise ValueError('state is not its source message')
                provenance = response.get('_guided_division_provenance')
                if not isinstance(provenance, dict) or set(provenance) != {'action', 'previous_session_id', 'previous_message_id'}:
                    raise ValueError('missing guided provenance')
                action, text = validate_action(provenance['action'])
                if action['operation'] not in GUIDED_OPERATIONS or text != expected['content']:
                    raise ValueError('action differs from literal source')
                source_ids.append(str(expected['source_id']))
                if action['operation'] == 'division_guided_start':
                    if provenance['previous_session_id'] is not None or provenance['previous_message_id'] is not None:
                        raise ValueError('start has an unexpected predecessor')
                    if state != start_state(action['dividend'], action['divisor'], state_id=key[1]):
                        raise ValueError('incorrect initial cursor')
                    if response.get('guided_assessment') != {'status':'not_assessed', 'competence_assessed':False, 'independent_work':'unknown'}:
                        raise ValueError('incorrect initial assessment')
                    break
                previous_session = provenance['previous_session_id']
                previous_message = provenance['previous_message_id']
                if action['expected_state_id'] != previous_message or not isinstance(previous_session, str):
                    raise ValueError('invalid predecessor identity')
                if (previous_session != key[0] and action['operation'] != 'division_guided_resume'
                        or action.get('source_learning_session_id') not in (None, previous_session)):
                    raise ValueError('invalid cross-session predecessor')
                session(previous_session)
                cur.execute('''SELECT m.*,s.content,s.author_role FROM teachcopilot.learning_messages m
                    JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
                    WHERE m.learning_session_id=%s AND m.client_message_id=%s AND m.learner_id=%s''',
                            (previous_session, previous_message, learner_id))
                previous = cur.fetchone()
                if previous is None:
                    raise ValueError('missing predecessor')
                previous = dict(previous)
                actual, assessment = transition(previous['response']['guided_division'], action['operation'],
                    state_id=key[1], value=action.get('value'))
                if actual != state or response.get('guided_assessment') != assessment:
                    raise ValueError('saved transition is not reproducible')
                expected = previous
            else:
                raise ValueError('guided chain exceeds verification bound')
            cur.execute('''SELECT 1 FROM teachcopilot.memory_sources WHERE learner_id=%s
                AND corrects_source_id::text=ANY(%s) LIMIT 1''', (learner_id, source_ids))
            if cur.fetchone():
                return dict(unavailable, reason='corrected_guided_source')
        except (ValueError, TypeError, KeyError):
            return dict(unavailable, reason='invalid_guided_chain')
        state = validate_state(candidate['response']['guided_division'])
        if state['status'] == 'completed':
            return dict(unavailable, reason='exercise_completed', guided_division=state)
        source_session = str(candidate['learning_session_id'])
    if origin == 'approved_memory':
        from .memory_service import approved_chat_memory
        memory = approved_chat_memory(store.get_learner_memory(learner_id, topic_id), topic_id)
        selected_ids = {str(row.get('source_id')) for row in memory.get('sources', [])}
        if (candidate['author_role'] != 'student' or not memory.get('version_id')
                or memory.get('stale') or str(candidate['source_id']) not in selected_ids):
            return dict(unavailable, reason='teacher_review_required')
    return {**{key:value for key,value in unavailable.items() if key == 'active_learning_session_id'},
            'available': True, 'origin': origin, 'guided_division': deepcopy(state),
            'resume_required': conversation_continued,
            **({'resume_reason': 'conversation_continued'} if conversation_continued else {}),
            'source_learning_session_id': source_session, 'prompt': render_prompt(state)}


def guided_response(store, learner_id, topic_id, actor_id, session_id, message_id, action, request_text):
    from .guided_division import action_response
    from .store import RevisionConflict
    if topic_id != TOPIC:
        raise ValueError('Пошаговое деление доступно в теме деления на однозначное и двузначное число.')
    previous_session = previous_message = None
    if action['operation'] == 'division_guided_start':
        state = start_state(action['dividend'], action['divisor'], state_id=message_id)
        assessment = {'status': 'not_assessed', 'competence_assessed': False, 'independent_work': 'unknown'}
    else:
        context = guided_context(store, learner_id, topic_id, actor_id, learning_session_id=session_id,
                                 pending_client_message_id=message_id)
        if not context.get('available') or context['guided_division']['state_id'] != action['expected_state_id']:
            raise RevisionConflict('guided division cursor changed; reload the current step')
        if context.get('resume_required') and action['operation'] != 'division_guided_resume':
            raise RevisionConflict('explicit saved step resume is required after conversation')
        if context['origin'] == 'approved_memory':
            if (action['operation'] != 'division_guided_resume'
                    or action.get('source_learning_session_id') != context['source_learning_session_id']):
                raise RevisionConflict('explicit reviewed lesson resume is required')
        elif action.get('source_learning_session_id') not in (None, context['source_learning_session_id']):
            raise RevisionConflict('source lesson changed')
        previous_session = context['source_learning_session_id']
        previous_message = action['expected_state_id']
        state, assessment = transition(context['guided_division'], action['operation'],
                                       state_id=message_id, value=action.get('value'))
    result = action_response(state, assessment, request_text, action['operation'])
    result['_guided_division_provenance'] = {'action': deepcopy(action),
        'previous_session_id': previous_session, 'previous_message_id': previous_message}
    return result
