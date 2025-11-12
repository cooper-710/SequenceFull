/**
 * Enhanced Dropdown Component
 * Converts all native select elements to custom styled dropdowns
 */
(function() {
    'use strict';

    // Initialize all dropdowns on page load
    document.addEventListener('DOMContentLoaded', function() {
        enhanceAllSelects();
    });

    // Also enhance dynamically added selects
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    if (node.tagName === 'SELECT' && !node.closest('.custom-select-wrapper')) {
                        enhanceSelect(node);
                    } else {
                        const selects = node.querySelectorAll && node.querySelectorAll('select:not(.custom-select-wrapper select)');
                        if (selects) {
                            selects.forEach(enhanceSelect);
                        }
                    }
                }
            });
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    function enhanceAllSelects() {
        // Select all single-select dropdowns, excluding multiple selects and already enhanced ones
        const selects = document.querySelectorAll('select:not([multiple]):not(.custom-select-wrapper select)');
        selects.forEach(function(select) {
            // Skip if it's inside a custom wrapper or is a multiple select
            if (!select.closest('.custom-select-wrapper') && !select.multiple) {
                enhanceSelect(select);
            }
        });
    }

    function enhanceSelect(select) {
        // Skip if already enhanced or if it's a multiple select
        if (select.closest('.custom-select-wrapper') || select.multiple) {
            return;
        }

        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'custom-select-wrapper';
        
        // Create custom select button
        const customSelect = document.createElement('div');
        customSelect.className = 'custom-select';
        
        const selectValue = document.createElement('span');
        selectValue.className = 'select-value';
        
        const selectArrow = document.createElement('span');
        selectArrow.className = 'select-arrow';
        selectArrow.innerHTML = '<i class="fas fa-chevron-down"></i>';
        
        customSelect.appendChild(selectValue);
        customSelect.appendChild(selectArrow);
        
        // Create dropdown
        const dropdown = document.createElement('div');
        dropdown.className = 'custom-select-dropdown';
        
        // Function to populate dropdown options
        function populateDropdown() {
            dropdown.innerHTML = '';
            const options = Array.from(select.options);
            options.forEach(function(option, index) {
                const customOption = document.createElement('div');
                customOption.className = 'custom-select-option';
                if (option.selected) {
                    customOption.classList.add('selected');
                }
                
                const check = document.createElement('span');
                check.className = 'option-check';
                check.innerHTML = '<i class="fas fa-check"></i>';
                
                const text = document.createElement('span');
                text.className = 'option-text';
                text.textContent = option.text;
                
                customOption.appendChild(check);
                customOption.appendChild(text);
                
                customOption.addEventListener('click', function(e) {
                    e.stopPropagation();
                    
                    // Update native select
                    select.selectedIndex = index;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    
                    // Update custom select
                    updateCustomSelect(select, customSelect, dropdown);
                    
                    // Close dropdown
                    closeDropdown(customSelect, dropdown);
                });
                
                dropdown.appendChild(customOption);
            });
        }
        
        // Initial population
        populateDropdown();
        
        // Listen for programmatic changes to the select
        select.addEventListener('change', function() {
            updateCustomSelect(select, customSelect, dropdown);
        });
        
        // Watch for option changes and rebuild dropdown if needed
        const optionObserver = new MutationObserver(function() {
            populateDropdown();
            updateCustomSelect(select, customSelect, dropdown);
        });
        
        optionObserver.observe(select, { childList: true, subtree: true });
        
        // Insert wrapper before select
        select.parentNode.insertBefore(wrapper, select);
        
        // Move select into wrapper (but keep it hidden)
        wrapper.appendChild(select);
        wrapper.appendChild(customSelect);
        wrapper.appendChild(dropdown);
        
        // Initialize display
        if (select.id === 'general_theme') {
            const currentTheme = document.body.dataset.theme;
            if (currentTheme && Array.from(select.options).some(opt => opt.value === currentTheme)) {
                select.value = currentTheme;
            }
        }
        updateCustomSelect(select, customSelect, dropdown);
        
        // Toggle dropdown
        customSelect.addEventListener('click', function(e) {
            e.stopPropagation();
            const isOpen = customSelect.classList.contains('open');
            
            // Close all other dropdowns
            closeAllDropdowns();

            // Sync specialized selects before opening
            if (select.id === 'general_theme') {
                const currentTheme = document.body.dataset.theme;
                if (currentTheme) {
                    const matchingOption = Array.from(select.options).find(option => option.value === currentTheme);
                    if (matchingOption && select.value !== currentTheme) {
                        select.value = currentTheme;
                        updateCustomSelect(select, customSelect, dropdown);
                    }
                }
            }
            
            if (!isOpen) {
                openDropdown(customSelect, dropdown);
            }
        });
        
        // Close on outside click
        document.addEventListener('click', function(e) {
            if (!wrapper.contains(e.target)) {
                closeDropdown(customSelect, dropdown);
            }
        });
        
        // Handle keyboard navigation
        customSelect.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const isOpen = customSelect.classList.contains('open');
                if (!isOpen) {
                    openDropdown(customSelect, dropdown);
                }
            } else if (e.key === 'Escape') {
                closeDropdown(customSelect, dropdown);
            }
        });
        
        dropdown.addEventListener('keydown', function(e) {
            const options = Array.from(dropdown.querySelectorAll('.custom-select-option'));
            const currentIndex = options.findIndex(opt => opt.classList.contains('selected'));
            let newIndex = currentIndex;
            
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                newIndex = (currentIndex + 1) % options.length;
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                newIndex = currentIndex <= 0 ? options.length - 1 : currentIndex - 1;
            } else if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (currentIndex >= 0) {
                    options[currentIndex].click();
                }
                return;
            } else if (e.key === 'Escape') {
                e.preventDefault();
                closeDropdown(customSelect, dropdown);
                return;
            }
            
            if (newIndex !== currentIndex) {
                options.forEach(opt => opt.classList.remove('selected'));
                options[newIndex].classList.add('selected');
                options[newIndex].scrollIntoView({ block: 'nearest' });
            }
        });

        // Allow external syncs without triggering change events
        select.addEventListener('custom-select:sync', function() {
            updateCustomSelect(select, customSelect, dropdown);
        });
    }

    function updateCustomSelect(select, customSelect, dropdown) {
        const selectedOption = select.options[select.selectedIndex];
        const selectValue = customSelect.querySelector('.select-value');
        
        if (selectedOption) {
            selectValue.textContent = selectedOption.text;
        }
        
        // Update selected state in dropdown
        const options = dropdown.querySelectorAll('.custom-select-option');
        options.forEach(function(option, index) {
            if (index === select.selectedIndex) {
                option.classList.add('selected');
            } else {
                option.classList.remove('selected');
            }
        });
    }

    function openDropdown(customSelect, dropdown) {
        customSelect.classList.add('open');
        dropdown.classList.add('open');
        
        // Scroll selected option into view
        const selectedOption = dropdown.querySelector('.custom-select-option.selected');
        if (selectedOption) {
            selectedOption.scrollIntoView({ block: 'nearest' });
        }
    }

    function closeDropdown(customSelect, dropdown) {
        customSelect.classList.remove('open');
        dropdown.classList.remove('open');
    }

    function closeAllDropdowns() {
        const openDropdowns = document.querySelectorAll('.custom-select.open');
        openDropdowns.forEach(function(customSelect) {
            const dropdown = customSelect.nextElementSibling;
            if (dropdown && dropdown.classList.contains('custom-select-dropdown')) {
                closeDropdown(customSelect, dropdown);
            }
        });
    }
})();

