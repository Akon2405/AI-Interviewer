class AudioVisualizer:
    @staticmethod
    def get_css():
        """Returns the CSS styling for the dynamic circular animations."""
        return """
        <style>
            .viz-container {
                display: flex; 
                justify-content: center; 
                align-items: center; 
                height: 300px;
                position: relative;
            }
            
            /* The base styling for the glowing rings */
            .ring {
                position: absolute;
                border: 2px solid rgba(255, 255, 255, 0.8);
                box-shadow: 0 0 15px #8a2be2, inset 0 0 15px #8a2be2; /* Deep purple glow */
            }

            /* IDLE STATE: A dim, perfectly round, quiet ring */
            .idle-ring {
                width: 280px;
                height: 280px;
                border-radius: 50%;
                opacity: 0.3;
                box-shadow: 0 0 5px #8a2be2, inset 0 0 5px #8a2be2;
                border: 1px solid rgba(255, 255, 255, 0.3);
                transition: all 0.5s ease;
            }

            /* ACTIVE STATE: Wobbly, spinning rings that simulate a sound wave */
            .active-ring-1 {
                width: 300px; height: 300px;
                border-radius: 45% 55% 40% 60% / 55% 45% 60% 40%;
                animation: spin 3s linear infinite;
            }
            .active-ring-2 {
                width: 295px; height: 295px;
                border: 2px solid rgba(255, 255, 255, 0.4);
                border-radius: 50% 50% 60% 40% / 40% 60% 50% 50%;
                animation: spin 2s linear infinite reverse;
            }
            .active-ring-3 {
                width: 310px; height: 310px;
                border: 1px solid rgba(138, 43, 226, 0.8);
                border-radius: 60% 40% 50% 50% / 50% 50% 40% 60%;
                animation: spin 4s linear infinite, pulse 1s ease-in-out infinite alternate;
                box-shadow: none;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes pulse {
                0% { transform: scale(0.95) rotate(0deg); opacity: 0.7;}
                100% { transform: scale(1.05) rotate(180deg); opacity: 1;}
            }
        </style>
        """

    @staticmethod
    def get_wave_html(is_speaking: bool):
        """Returns the HTML state for the circle depending on if the Professor is talking."""
        if is_speaking:
            # Active spinning, morphing rings
            return """
            <div class="viz-container">
                <div class="ring active-ring-1"></div>
                <div class="ring active-ring-2"></div>
                <div class="ring active-ring-3"></div>
            </div>
            """
        else:
            # Idle static ring
            return """
            <div class="viz-container">
                <div class="idle-ring"></div>
            </div>
            """