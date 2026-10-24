// Keep collapsible tool-call boxes (<details>) open across Dash re-renders: the chat's poll
// re-renders the transcript every second while a turn streams, which would otherwise reset
// each box. Open state is tracked per box position and re-applied on renders that don't
// change the conversation; a new user message (new turn, or a session switch) resets it.
(function () {
    var open = new Map();
    var lastUserMsgs = -1;
    document.addEventListener('toggle', function (e) {
        var chat = document.getElementById('chat');
        if (!chat || !chat.contains(e.target)) return;
        var i = Array.prototype.indexOf.call(chat.querySelectorAll('details'), e.target);
        if (i >= 0) open.set(i, e.target.open);
    }, true);
    new MutationObserver(function () {
        var chat = document.getElementById('chat');
        if (!chat) return;
        var userMsgs = chat.querySelectorAll('.msg-user').length;
        if (userMsgs !== lastUserMsgs) {
            lastUserMsgs = userMsgs;
            open.clear();
            return;
        }
        chat.querySelectorAll('details').forEach(function (d, i) {
            if (open.has(i) && d.open !== open.get(i)) d.open = open.get(i);
        });
    }).observe(document.documentElement, {childList: true, subtree: true});
})();
