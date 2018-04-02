<template>
    <v-container fill-height pa-0>
        <v-layout column fill-height :style="'height:100%;max-height:' + maxHeight">
            <v-toolbar>
                <v-toolbar-title>
                    LyaBot
                </v-toolbar-title>
            </v-toolbar>
            <v-flex ref="scroller" style="overflow-y:auto;" class="grey darken-3">
                <v-layout column d-flex>
                    <v-flex style="flex:2"/> <!-- spacer -->
                    <v-flex v-for="(item, index) in items" :key="index" :class="'chatMessage' + (item.fromLya ? ' fromLya' : '')">
                        <v-layout column>
                            <v-layout row d-flex pa-2>
                                <v-flex class="avatar" ma-2>
                                      <img v-if="item.fromLya" src="https://upload.wikimedia.org/wikipedia/commons/b/b0/Noto_Emoji_Oreo_1f916.svg">
                                      <img v-else src="http://www.xn--icne-wqa.com/images/icones/1/2/animals-cat.png">
                                </v-flex>
                                <v-flex class="content" ma-2>
                                    <v-layout d-flex column>
                                        <v-flex>{{ item.data }}</v-flex>
                                        <v-flex class="status grey--text lighten-5"  v-moment-ago="item.datetime"></v-flex>
                                    </v-layout>
                                </v-flex>
                            </v-layout>
                            <v-divider v-if="index < items.length -1"></v-divider>
                        </v-layout>
                    </v-flex>
                    <v-flex v-if="lyaIsTyping" id="isTyping">
                        <v-divider/>
                        <v-layout pa-2>
                            Lya is typing{{ threeDotLoader }}
                        </v-layout>
                    </v-flex>
                </v-layout>
            </v-flex>
            <v-flex class="grey lighten-2" pa-2 style="max-height:90px;height:90px;border-radius:0 0 3px 3px;justify-self:flex-end;">
                <v-text-field
                    color="grey darken-4"
                    v-model="input"
                    @keyup.native.enter="send()"
                    label="Say something to Lya" light>
                </v-text-field>
            </v-flex>
        </v-layout>
    </v-container>
</template>

<style scoped>
#isTyping {
    font-style:italic;
    flex-grow:0 !important;
    font-size:80%;
}
.chatMessage {
    flex:0 0 auto !important; /* Don't grow, don't shrink. Fit size. */
    overflow:hidden;
    text-align:right;
}
    .avatar {
        max-width:40px;
        max-height:40px;
        order:1;
    }
    .status {
        font-style:italic;
        font-size:80%;
    }
    .content {
        order:0;
        justify-content:flex-end;
    }

.chatMessage.fromLya {
    text-align:left;
}
    .chatMessage.fromLya .avatar {
        order:0;
    }

    .chatMessage.fromLya .content {
        order:1;
        justify-content:flex-start;
    }

.reveal-enter-active, .reveal-leave-active {
    transition: all 50s ease;
    max-height:0;
}
.reveal-enter, .reveal-leave-to /* .reveal-leave-active below version 2.1.8 */ {
    max-height:500px;
}
</style>

<script>
import axios from 'axios';
import Config from "@/Config.json"

export default {
    data () {
        return {
            maxHeight: "",
            input: "",
            lyaIsTyping: null,
            threeDotLoader: "",
            items: [],
        }
    },
    beforeMount: function () {
        window.addEventListener('resize', this.handleResize);
        this.handleResize();
    },
    beforeDestroy: function () {
       window.removeEventListener('resize', this.handleResize);
    },
    methods: {
        scroll() {
            setTimeout(() => {
                console.log(this.$refs.scroller.scrollTop, this.$refs.scroller.scrollHeight);
                this.$refs.scroller.scrollTop = this.$refs.scroller.scrollHeight + 100;
            }, 1);
        },

        handleResize() {
            // Limit chat window height to window height (so only the conversation scroll)
            this.maxHeight = document.documentElement.clientHeight + "px";
        },

        isTyping(isTyping=true) {
            if (isTyping) {
                if (this.lyaIsTyping !== null) return;
                this.lyaIsTyping = true;
                this.threeDotLoader = ".";
                this.threeDotLoaderTimer = setInterval(() => {
                    this.threeDotLoader += ".";
                    if (this.threeDotLoader.length > 3) {
                        this.threeDotLoader = "";
                    }
                }, 220);
            } else {
                clearTimeout(this.threeDotLoaderTimer);
                this.lyaIsTyping = false;
            }
        },

        send() {
            const item = { datetime: Date.now(), data: this.input };
            this.items.push(item);

            this.lyaIsTyping = null;
            setTimeout(() => {
                this.isTyping();
                this.scroll();
            }, 1);

            const start = Date.now();
            axios.get(Config.BOT_URL, { params: { question: encodeURIComponent(this.input)} })
                .then((res) => {
                    const diff = Config.MIN_REPLY_TIME - (Date.now() - start);
                    setTimeout(() => {
                        this.isTyping(false);
                        this.items.push({ datetime: Date.now(), data: res.data.replies[0], fromLya: true });
                        this.scroll();
                    }, diff);
                }).catch((e) => {
                    item.apiError = `Failed to send message: ${e.message}`;
                    this.isTyping(false);
                });

            this.input = "";
        }
    }
}
</script>
